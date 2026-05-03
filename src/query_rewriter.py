"""多轮对话检索改写模块。

这个模块只负责一件事：在用户发起省略式追问时，把“依赖历史才能理解”的问题
改写成“脱离上下文也能直接送去检索”的独立问题。

注意这里的改写只服务检索，不会改变：
1. 用户在前端看到的原始问题。
2. 生成阶段最终交给回答模型的原始问题文本。
3. 对话历史里保存的消息内容。
"""

from __future__ import annotations

import httpx
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

from src.config import (
    ARK_API_KEY,
    MAX_QUERY_CHARS,
    QUERY_REWRITE_ENABLED,
    QUERY_REWRITE_MAX_HISTORY_MESSAGES,
    QUERY_REWRITE_MODEL_NAME,
    QUERY_REWRITE_TIMEOUT_MS,
    QUERY_REWRITE_TRIGGER_MAX_CHARS,
)

_REFERENTIAL_QUERY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(那|那么|那这|那它|那这个|那第二款|那第二个)",
        r"^(它|这个|这个产品|该产品|该方案|该功能)",
        r"^(第二款|第二个|前者|后者)",
        r"(上面提到的|刚才提到的|前面提到的|上一轮提到的)",
        r"(它的|它们的|其优点|其缺点|其限制|其适用场景)",
    ]
]

_SHORT_FOLLOW_UP_KEYWORDS = {
    "优点",
    "缺点",
    "限制",
    "不足",
    "区别",
    "参数",
    "价格",
    "特点",
    "适用场景",
    "适合谁",
}

_EXPLICIT_STANDALONE_QUERY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(请总结|总结一下|请概括|概括一下)",
        r"^(请介绍|介绍一下|请解释|解释一下|说明一下)",
        r"^(什么是|如何|为什么|哪些|哪几款|哪几种|怎么做)",
    ]
]

_COMPOUND_QUERY_SPLIT_PATTERN = re.compile(r"[？?；;\n]+")


def _build_client(timeout_ms: int) -> OpenAI:
    timeout_seconds = max(0.5, timeout_ms / 1000)
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
        timeout=httpx.Timeout(timeout=timeout_seconds, connect=min(timeout_seconds, 1.5)),
        max_retries=0,
    )


def _normalize_query_text(query: str) -> str:
    return re.sub(r"[\s\.,!?，。！？、:：;；\"'“”‘’()（）【】\[\]<>《》]+", "", (query or "").lower()).strip()


def _split_possible_compound_query(query: str) -> List[str]:
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized_query:
        return []

    raw_parts = [
        part.strip(" \t\r\n?？;；")
        for part in _COMPOUND_QUERY_SPLIT_PATTERN.split(normalized_query)
    ]

    deduped_parts: List[str] = []
    seen_parts = set()
    for part in raw_parts:
        if not part:
            continue
        normalized_part = _normalize_query_text(part)
        if not normalized_part or normalized_part in seen_parts:
            continue
        seen_parts.add(normalized_part)
        deduped_parts.append(part)

    return deduped_parts or [normalized_query]


def _is_single_question_query(query: str) -> bool:
    return len(_split_possible_compound_query(query)) <= 1


def _trim_history(chat_history: List[dict] | None, max_messages: int) -> List[dict]:
    if not chat_history:
        return []

    recent_messages = [
        {"role": str(item.get("role", "")).strip(), "content": str(item.get("content", "")).strip()}
        for item in chat_history[-max_messages:]
        if str(item.get("role", "")).strip() in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]
    return recent_messages


def _has_assistant_context(chat_history: List[dict]) -> bool:
    return any(item.get("role") == "assistant" for item in chat_history)


def _should_rewrite_with_trimmed_history(original_query: str, trimmed_history: List[dict]) -> bool:
    normalized_query = _normalize_query_text(original_query)
    if not normalized_query:
        return False
    if not trimmed_history:
        return False
    if not _has_assistant_context(trimmed_history):
        return False

    query_text = original_query.strip()

    for pattern in _REFERENTIAL_QUERY_PATTERNS:
        if pattern.search(query_text):
            return True

    # 明显完整且较长的问题，大概率已经能直接检索，不值得再额外做一次重写。
    if len(query_text) > QUERY_REWRITE_TRIGGER_MAX_CHARS * 2:
        return False

    # 对语义已经完整的问题前缀做快速跳过，降低无收益的模型调用。
    for pattern in _EXPLICIT_STANDALONE_QUERY_PATTERNS:
        if pattern.search(query_text):
            return False

    # 没有明显指代词时，只对特别短且明显像“追问槽位”的问题启用改写，
    # 避免把本来就能直接检索的问题也多做一次轻量模型调用。
    if len(normalized_query) <= max(8, QUERY_REWRITE_TRIGGER_MAX_CHARS // 4):
        return any(keyword in query_text for keyword in _SHORT_FOLLOW_UP_KEYWORDS)

    return False


def should_rewrite_query(original_query: str, chat_history: List[dict] | None = None) -> bool:
    """判断当前问题是否值得做多轮上下文改写。"""
    if not QUERY_REWRITE_ENABLED:
        return False

    trimmed_history = _trim_history(chat_history, QUERY_REWRITE_MAX_HISTORY_MESSAGES)
    return _should_rewrite_with_trimmed_history(original_query, trimmed_history)


def _format_history_for_prompt(chat_history: List[dict]) -> str:
    lines = []
    for item in chat_history:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}: {item.get('content', '').strip()}")
    return "\n".join(lines)


def _latest_message_content(chat_history: List[dict], role: str) -> str:
    for item in reversed(chat_history):
        if item.get("role") == role and item.get("content", "").strip():
            return str(item.get("content", "")).strip()
    return ""


def _sanitize_context_anchor_text(text: str) -> str:
    cleaned = re.sub(r"\[[A-Za-z]\d+\]", " ", text or "")
    cleaned = re.sub(r"[？?；;\n]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip("。.!！,，、:：;；")


def _build_local_fallback_query(original_query: str, chat_history: List[dict]) -> str:
    query_text = original_query.strip()
    if not query_text:
        return original_query
    if not any(pattern.search(query_text) for pattern in _REFERENTIAL_QUERY_PATTERNS):
        return original_query

    # 优先拿上一轮助手回答做锚点，避免把上一条用户问句原样拼进当前追问，
    # 从而制造出“两个问号 -> 两个子问题”的伪 compound query。
    anchor_text = _latest_message_content(chat_history, "assistant") or _latest_message_content(chat_history, "user")
    sanitized_anchor = _sanitize_context_anchor_text(anchor_text)
    if not sanitized_anchor:
        return original_query

    clipped_anchor = sanitized_anchor[:160].rstrip()
    if not clipped_anchor:
        return original_query

    candidate = _sanitize_rewritten_query(f"{clipped_anchor} {query_text}", original_query)
    if not _is_single_question_query(candidate):
        return original_query
    if _normalize_query_text(candidate) == _normalize_query_text(original_query):
        return original_query
    return candidate


def _sanitize_rewritten_query(rewritten_query: str, original_query: str) -> str:
    cleaned = (rewritten_query or "").strip()
    if not cleaned:
        return original_query

    cleaned = cleaned.strip("`\"'“”‘’ ")
    cleaned = re.sub(r"^(改写后问题|改写问题|独立检索问题|重写问题|问题)[:：]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return original_query
    if len(cleaned) > MAX_QUERY_CHARS:
        return original_query

    # 轻量模型偶尔会输出多行解释，这里只保留第一条真正的问题文本。
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    first_line = first_line.strip("`\"'“”‘’ ")
    if not first_line:
        return original_query
    if not _is_single_question_query(first_line):
        return original_query
    return first_line


def rewrite_query_with_context(
    original_query: str,
    chat_history: List[dict] | None = None,
    max_history_messages: int | None = None,
) -> Dict[str, Any]:
    """结合多轮历史，把当前追问改写成可直接检索的独立问题。"""
    start = time.perf_counter()
    max_messages = max_history_messages or QUERY_REWRITE_MAX_HISTORY_MESSAGES
    trimmed_history = _trim_history(chat_history, max_messages)
    base_result: Dict[str, Any] = {
        "applied": False,
        "rewritten_query": original_query,
        "reason": "skipped",
        "timing_ms": 0.0,
        "history_messages_used": len(trimmed_history),
        "model": QUERY_REWRITE_MODEL_NAME,
    }

    if not QUERY_REWRITE_ENABLED:
        base_result["reason"] = "disabled"
        return base_result
    if not ARK_API_KEY:
        base_result["reason"] = "missing_api_key"
        return base_result
    if not trimmed_history:
        base_result["reason"] = "no_history"
        return base_result
    if not _has_assistant_context(trimmed_history):
        base_result["reason"] = "no_assistant_context"
        return base_result
    if not _should_rewrite_with_trimmed_history(original_query, trimmed_history):
        base_result["reason"] = "heuristic_skip"
        return base_result

    system_prompt = (
        "你是一个面向检索的查询改写器。"
        "你的任务不是回答问题，而是把依赖上下文的追问改写成一条可独立检索的完整问题。"
        "改写规则：\n"
        "1. 只补全历史中明确出现的对象、产品、主题、文档上下文，不要编造新事实。\n"
        "2. 如果历史不足以补全，就原样返回当前问题。\n"
        "3. 输出只包含改写后的单句问题，不要解释，不要加前缀，不要输出多个问句、分号或换行。\n"
        "4. 保持原问题语言风格，默认使用简体中文。"
    )
    user_prompt = (
        "请基于下面的对话历史，把当前用户问题改写成一条可以直接送去文档检索的完整问题。\n\n"
        f"对话历史：\n{_format_history_for_prompt(trimmed_history)}\n\n"
        f"当前用户问题：{original_query.strip()}"
    )

    request_kwargs: Dict[str, Any] = {
        "model": QUERY_REWRITE_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 80,
        "temperature": 0.0,
    }

    client = _build_client(timeout_ms=QUERY_REWRITE_TIMEOUT_MS)
    try:
        response = client.chat.completions.create(**request_kwargs)
        raw_text = response.choices[0].message.content.strip() if response.choices else original_query
        rewritten_query = _sanitize_rewritten_query(raw_text, original_query)
        base_result["rewritten_query"] = rewritten_query
        base_result["applied"] = _normalize_query_text(rewritten_query) != _normalize_query_text(original_query)
        base_result["reason"] = "rewritten" if base_result["applied"] else "unchanged"
    except Exception as exc:
        fallback_query = _build_local_fallback_query(original_query, trimmed_history)
        if _normalize_query_text(fallback_query) != _normalize_query_text(original_query):
            base_result["rewritten_query"] = fallback_query
            base_result["applied"] = True
            base_result["reason"] = f"fallback_local:{exc.__class__.__name__}"
        else:
            base_result["reason"] = f"error:{exc.__class__.__name__}"
    finally:
        base_result["timing_ms"] = round((time.perf_counter() - start) * 1000, 2)

    return base_result