"""Automatic faithfulness judging for multimodal RAG evaluation.

当前项目的证据是“页面图像 + 页码/文档元数据”，而不是纯文本 context。
因此这里不直接套用 text-only 的 RAG faithfulness 流程，而是复用现有的
OpenAI 兼容多模态模型，对答案是否被证据页支持做自动判别。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from src.config import ARK_API_KEY, DOUBAO_MODEL_NAME


DEFAULT_AUTOMATIC_FAITHFULNESS_MODEL_NAME = os.getenv(
    "AUTOMATIC_FAITHFULNESS_MODEL_NAME",
    DOUBAO_MODEL_NAME,
)
DEFAULT_AUTOMATIC_FAITHFULNESS_MAX_IMAGES = max(
    1,
    int(os.getenv("AUTOMATIC_FAITHFULNESS_MAX_IMAGES", "5")),
)
DEFAULT_AUTOMATIC_FAITHFULNESS_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("AUTOMATIC_FAITHFULNESS_TIMEOUT_SECONDS", "30")),
)
_PROVIDER_NAME = "doubao_multimodal_judge"


def _build_client() -> OpenAI:
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )


def _build_evidence_metadata_text(evidences: Sequence[Dict[str, Any]], limit: int) -> str:
    lines: List[str] = []
    for index, evidence in enumerate(evidences[:limit], start=1):
        evidence_id = str(evidence.get("evidence_id") or f"E{index}")
        document_name = str(evidence.get("document_name") or evidence.get("document_id") or "unknown")
        page_number = int(evidence.get("page_number") or 0)
        score = evidence.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"[{evidence_id}] 文档={document_name} 页码={page_number} 检索分数={float(score):.2f}")
        else:
            lines.append(f"[{evidence_id}] 文档={document_name} 页码={page_number}")
    return "\n".join(lines)


def _normalize_image_data_url(raw_image_value: str) -> str:
    cleaned = re.sub(r"\s+", "", raw_image_value.strip())
    if cleaned.startswith("data:image"):
        return cleaned
    return f"data:image/png;base64,{cleaned}"


def _build_image_content(evidences: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for evidence in evidences[:limit]:
        image_base64 = str(evidence.get("image_base64") or "").strip()
        if not image_base64:
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _normalize_image_data_url(image_base64)},
            }
        )
    return content


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _derive_label(score: Optional[float]) -> str:
    if score is None:
        return "unavailable"
    if score >= 0.85:
        return "supported"
    if score >= 0.35:
        return "partially_supported"
    return "unsupported"


def _normalize_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_score = payload.get("score")
    score: Optional[float]
    if raw_score in (None, ""):
        score = None
    else:
        score = max(0.0, min(1.0, round(float(raw_score), 4)))

    label = str(payload.get("label") or "").strip() or _derive_label(score)
    reason = str(payload.get("reason") or "").strip()
    return {
        "score": score,
        "label": label,
        "reason": reason,
        "provider": _PROVIDER_NAME,
    }


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fence(raw_text)
    try:
        return _normalize_result(json.loads(cleaned))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return _normalize_result(json.loads(match.group(0)))


def score_multimodal_faithfulness(
    *,
    question: str,
    answer_text: str,
    evidences: Sequence[Dict[str, Any]],
    model_name: Optional[str] = None,
    max_images: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    if not str(answer_text or "").strip():
        return {
            "score": None,
            "label": "skipped_empty_answer",
            "reason": "Answer text is empty.",
            "provider": _PROVIDER_NAME,
        }

    if not evidences:
        return {
            "score": None,
            "label": "skipped_no_evidence",
            "reason": "No returned evidences available for judging.",
            "provider": _PROVIDER_NAME,
        }

    if not ARK_API_KEY:
        return {
            "score": None,
            "label": "unavailable_missing_api_key",
            "reason": "ARK_API_KEY is not configured.",
            "provider": _PROVIDER_NAME,
        }

    image_limit = max_images or DEFAULT_AUTOMATIC_FAITHFULNESS_MAX_IMAGES
    timeout = timeout_seconds or DEFAULT_AUTOMATIC_FAITHFULNESS_TIMEOUT_SECONDS
    image_content = _build_image_content(evidences, image_limit)
    if not image_content:
        return {
            "score": None,
            "label": "skipped_no_images",
            "reason": "No evidence images available for judging.",
            "provider": _PROVIDER_NAME,
        }

    metadata_text = _build_evidence_metadata_text(evidences, image_limit)
    user_prompt = (
        "请判断下面这条 RAG 回答是否被提供的证据页直接支持。\n"
        "要求：\n"
        "1. 只能依据给定证据图片和证据元数据判断，不能使用常识补全。\n"
        "2. 忽略回答里的引用格式是否漂亮，只判断事实内容是否被支持。\n"
        "3. 若回答大体正确但包含超出证据的补充、模糊推断或部分无法核验内容，给 0.5 左右。\n"
        "4. 若回答关键结论与证据矛盾，或大部分内容缺少证据支持，给 0.0。\n"
        "5. 若回答核心结论都能被证据直接支持，且没有明显超出证据的事实补全，给 1.0。\n"
        "6. 只输出 JSON，对象格式固定为："
        '{"score": 0.0-1.0, "label": "supported|partially_supported|unsupported", "reason": "一句简短中文说明"}。\n\n'
        f"问题：{question}\n\n"
        f"系统回答：{answer_text}\n\n"
        f"证据元数据：\n{metadata_text}"
    )

    try:
        response = _build_client().chat.completions.create(
            model=model_name or DEFAULT_AUTOMATIC_FAITHFULNESS_MODEL_NAME,
            temperature=0,
            timeout=timeout,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个严格的 RAG Faithfulness 评审器。"
                        "你只能根据当前证据判断回答是否被支持，不能使用外部知识补全。"
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}, *image_content],
                },
            ],
        )
        raw_text = str(response.choices[0].message.content or "")
        result = _parse_json_response(raw_text)
        result["reason"] = result["reason"] or "Judge returned no reason."
        return result
    except Exception as exc:  # pragma: no cover - network / remote model failures
        return {
            "score": None,
            "label": "judge_error",
            "reason": str(exc),
            "provider": _PROVIDER_NAME,
        }
