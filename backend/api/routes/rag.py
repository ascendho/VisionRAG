import asyncio
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
import os
import json
import re
import shutil
import tempfile
import glob
import time

from src.doc_processor import (
    get_file_hash,
    process_image_to_images,
    process_pdf_to_images,
    process_pptx_to_images,
    process_text_to_images,
)
from src.llm_generator import generate_answer_stream, generate_suggested_questions
from src.config import DEFAULT_MIN_SCORE, QUERY_GUARD_ENABLED
from src.query_rewriter import rewrite_query_with_context

router = APIRouter()
logger = logging.getLogger(__name__)

# DTO schema for chat request
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    top_k: int = 5
    min_score: float = DEFAULT_MIN_SCORE


_PRE_RETRIEVAL_GUARD_PATTERNS = [
    (
        "assistant_identity",
        re.compile(r"^(你是谁|你是誰|你叫什么|你叫啥|你叫什麼|介绍你自己|介绍一下你自己|请介绍一下你自己|你能做什么|你会什么)$", re.IGNORECASE),
    ),
    (
        "small_talk",
        re.compile(r"^(你好|您好|hello|hi|hey)$", re.IGNORECASE),
    ),
    (
        "off_topic_smalltalk",
        re.compile(r"^(天气怎么样|今天天气怎么样|讲个笑话|说个笑话)$", re.IGNORECASE),
    ),
]

_COMPOUND_QUERY_SPLIT_PATTERN = re.compile(r"[？?；;\n]+")
_NEAR_THRESHOLD_GAP = 0.05


def _normalize_query_text(query: str) -> str:
    return re.sub(r"[\s\.,!?，。！？、:：;；\"'“”‘’()（）【】\[\]<>《》]+", "", query.lower()).strip()


def _build_guarded_payload(reason: str, stage: str, message: Optional[str] = None, meta: Optional[dict] = None) -> dict:
    if message is None:
        if reason == "assistant_identity":
            message = (
                "这个问题不属于当前文档问答范围。"
                "我只能依据已上传文档中的证据回答问题，暂时不能回答关于助手身份或能力设定的问题。"
                "请改问与文档内容直接相关的问题。"
            )
        else:
            message = (
                "这个问题与当前文档内容范围无关，或缺少足够可靠的文档证据。"
                "我暂时不返回依据页，也不会基于外部常识作答。"
                "请换一个更贴近文档内容的问题。"
            )

    return {
        "message": message,
        "reason": reason,
        "stage": stage,
        "meta": meta or {},
    }


def _guard_obvious_out_of_scope(query: str) -> Optional[dict]:
    if not QUERY_GUARD_ENABLED:
        return None

    normalized = _normalize_query_text(query)
    if not normalized:
        return None

    for reason, pattern in _PRE_RETRIEVAL_GUARD_PATTERNS:
        if pattern.match(normalized):
            return _build_guarded_payload(reason=reason, stage="pre_retrieval")

    return None


def _split_compound_query(query: str) -> List[str]:
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


def _result_identity(result: Dict[str, Any]) -> tuple:
    return (
        result.get("document_id", ""),
        result.get("page_number", -1),
        result.get("image_path", ""),
    )


def _filter_results_by_min_score(results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    return [dict(item) for item in results if float(item.get("score", 0.0)) >= min_score]


def _prepare_selected_result(item: Dict[str, Any], min_score: float, *, is_fallback: bool, reason: str = "") -> Dict[str, Any]:
    cloned = dict(item)
    source_score = float(cloned.get("score", 0.0))
    fallback_gap = round(max(float(min_score) - source_score, 0.0), 2)
    cloned["fallback_below_threshold"] = is_fallback
    cloned["fallback_source_score"] = source_score
    cloned["fallback_gap_to_threshold"] = fallback_gap if is_fallback else 0.0
    cloned["fallback_tier"] = (
        "formal"
        if not is_fallback
        else "near_threshold" if fallback_gap <= _NEAR_THRESHOLD_GAP else "low_confidence"
    )
    cloned["fallback_reason"] = reason if is_fallback else ""
    return cloned


def _select_results_with_threshold_fallback(
    results: List[Dict[str, Any]],
    min_score: float,
    fallback_limit: int = 1,
    selection_limit: Optional[int] = None,
) -> Dict[str, Any]:
    max_selected = max(1, int(selection_limit or len(results) or fallback_limit))
    strict_results = _filter_results_by_min_score(results, min_score)[:max_selected]
    prepared_results = [
        _prepare_selected_result(item, min_score, is_fallback=False)
        for item in strict_results
    ]

    supplemental_results: List[Dict[str, Any]] = []
    available_slots = max(max_selected - len(prepared_results), 0)
    near_threshold_floor = max(float(min_score) - _NEAR_THRESHOLD_GAP, 0.0)
    if available_slots > 0:
        for item in results:
            score = float(item.get("score", 0.0))
            if score >= min_score or score < near_threshold_floor:
                continue
            supplemental_results.append(
                _prepare_selected_result(
                    item,
                    min_score,
                    is_fallback=True,
                    reason=f"supplemented_near_threshold<{min_score:.2f}",
                )
            )
            if len(supplemental_results) >= available_slots:
                break

    if prepared_results or supplemental_results:
        fallback_best_score = 0.0
        if supplemental_results:
            fallback_best_score = round(
                max(float(item.get("score", 0.0)) for item in supplemental_results),
                2,
            )
        return {
            "selected_results": (prepared_results + supplemental_results)[:max_selected],
            "fallback_used": bool(supplemental_results),
            "fallback_count": len(supplemental_results),
            "fallback_best_score": fallback_best_score,
        }

    if not results:
        return {
            "selected_results": [],
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_best_score": 0.0,
        }

    promoted_results: List[Dict[str, Any]] = []
    for item in results[:min(max_selected, max(1, int(fallback_limit)))]:
        promoted_results.append(
            _prepare_selected_result(
                item,
                min_score,
                is_fallback=True,
                reason=f"best_available_below_threshold<{min_score:.2f}",
            )
        )

    return {
        "selected_results": promoted_results,
        "fallback_used": True,
        "fallback_count": len(promoted_results),
        "fallback_best_score": round(float(promoted_results[0].get("score", 0.0)), 2),
    }


def _best_result_score(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return round(max(float(item.get("score", 0.0)) for item in results), 2)


def _build_insufficient_evidence_answer(min_score: float, sub_queries: List[str]) -> str:
    lines = [
        "### 结论",
        "",
        "根据当前证据无法确认。",
        "",
        "### 依据",
        "",
        f"当前检索到的候选页都未达到采用阈值 {min_score:.2f}，因此本轮不引用任何正式证据。",
    ]

    if sub_queries:
        lines.extend([
            "",
            "本轮涉及的问题：",
        ])
        lines.extend([f"- {sub_query}" for sub_query in sub_queries])

    lines.extend([
        "",
        "你可以尝试缩小问题范围、换一种更具体的问法，或临时降低阈值后再查看未采用候选页。",
    ])
    return "\n".join(lines)


def _sanitize_sub_query_context_text(query: str) -> str:
    cleaned = re.sub(r"[？?；;\n]+", " ", (query or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip("。.!！,，、:：;；")


def _build_compound_rewrite_history(chat_history: Optional[List[dict]], prior_sub_queries: List[str]) -> List[dict]:
    history = [
        {"role": str(item.get("role", "")).strip(), "content": str(item.get("content", "")).strip()}
        for item in (chat_history or [])
        if str(item.get("role", "")).strip() in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]
    if prior_sub_queries:
        latest_context = _sanitize_sub_query_context_text(prior_sub_queries[-1])
        if latest_context:
            # 这里刻意把上一子问题压成一条简短 assistant 上下文，
            # 让后续“那他/那它/第二个...”这类指代追问也能走已有 rewrite/fallback 逻辑。
            history.append({"role": "assistant", "content": latest_context})
    return history


def _normalize_anchor_query_for_bridge(query: str) -> str:
    cleaned = _sanitize_sub_query_context_text(query)
    cleaned = re.sub(r"(是什么|是啥|是谁|有哪些|有什么|有没有|是否|能否|可否|会不会|如何|怎么样|多少|几种|几个|几款|吗|呢|么|是)$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip("的")


def _build_compound_local_bridge_query(current_sub_query: str, prior_sub_queries: List[str]) -> str:
    query_text = _sanitize_sub_query_context_text(current_sub_query)
    if not query_text or not prior_sub_queries:
        return current_sub_query

    anchor_query = _normalize_anchor_query_for_bridge(prior_sub_queries[-1])
    if not anchor_query:
        return current_sub_query

    referential_replacements = [
        (r"^(那|那么)?他的", f"{anchor_query}的"),
        (r"^(那|那么)?她的", f"{anchor_query}的"),
        (r"^(那|那么)?它的", f"{anchor_query}的"),
        (r"^(那|那么)?其", f"{anchor_query}的"),
        (r"^(那|那么)?他", anchor_query),
        (r"^(那|那么)?她", anchor_query),
        (r"^(那|那么)?它", anchor_query),
    ]

    for pattern, replacement in referential_replacements:
        rewritten = re.sub(pattern, replacement, query_text, count=1)
        if rewritten != query_text:
            rewritten = re.sub(r"\s+", " ", rewritten).strip()
            return rewritten

    return current_sub_query


def _plan_retrieval_queries(original_query: str, chat_history: Optional[List[dict]]) -> Dict[str, Any]:
    query_labels = _split_compound_query(original_query)
    if len(query_labels) <= 1:
        rewrite_meta = rewrite_query_with_context(
            original_query=original_query,
            chat_history=chat_history,
        )
        retrieval_query = str(rewrite_meta.get("rewritten_query") or original_query)
        rewrite_meta["sub_query_rewrites"] = [
            {
                "original_query": original_query,
                "retrieval_query": retrieval_query,
                "applied": bool(rewrite_meta.get("applied")),
                "reason": str(rewrite_meta.get("reason") or "unknown"),
                "timing_ms": float(rewrite_meta.get("timing_ms") or 0.0),
                "history_messages_used": int(rewrite_meta.get("history_messages_used") or 0),
                "model": str(rewrite_meta.get("model") or ""),
            }
        ]
        return {
            "query_labels": query_labels or [original_query],
            "retrieval_sub_queries": [retrieval_query],
            "rewrite_meta": rewrite_meta,
        }

    retrieval_sub_queries: List[str] = []
    sub_query_rewrites: List[Dict[str, Any]] = []
    total_rewrite_ms = 0.0
    any_rewrite_applied = False
    max_history_messages_used = 0
    rewrite_model = ""

    for index, sub_query in enumerate(query_labels):
        bridged_query = _build_compound_local_bridge_query(sub_query, query_labels[:index])
        if bridged_query != sub_query:
            retrieval_query = bridged_query
            sub_meta = {
                "applied": True,
                "reason": "compound_local_context_bridge",
                "timing_ms": 0.0,
                "history_messages_used": 1,
                "model": "",
            }
        else:
            compound_history = _build_compound_rewrite_history(chat_history, query_labels[:index])
            sub_meta = rewrite_query_with_context(
                original_query=sub_query,
                chat_history=compound_history,
            )
            retrieval_query = str(sub_meta.get("rewritten_query") or sub_query)

        retrieval_sub_queries.append(retrieval_query)
        sub_query_rewrites.append(
            {
                "original_query": sub_query,
                "retrieval_query": retrieval_query,
                "applied": bool(sub_meta.get("applied")),
                "reason": str(sub_meta.get("reason") or "unknown"),
                "timing_ms": float(sub_meta.get("timing_ms") or 0.0),
                "history_messages_used": int(sub_meta.get("history_messages_used") or 0),
                "model": str(sub_meta.get("model") or ""),
            }
        )
        total_rewrite_ms += float(sub_meta.get("timing_ms") or 0.0)
        any_rewrite_applied = any_rewrite_applied or bool(sub_meta.get("applied"))
        max_history_messages_used = max(max_history_messages_used, int(sub_meta.get("history_messages_used") or 0))
        if not rewrite_model:
            rewrite_model = str(sub_meta.get("model") or "")

    return {
        "query_labels": query_labels,
        "retrieval_sub_queries": retrieval_sub_queries,
        "rewrite_meta": {
            "applied": any_rewrite_applied,
            "rewritten_query": original_query,
            "reason": "compound_subquery_rewrite",
            "timing_ms": round(total_rewrite_ms, 2),
            "history_messages_used": max_history_messages_used,
            "model": rewrite_model,
            "sub_query_rewrites": sub_query_rewrites,
        },
    }


def _apply_reusable_sub_query_support(
    vector_store_instance,
    selected_results: List[Dict[str, Any]],
    sub_query_support: List[Dict[str, Any]],
    unsupported_sub_queries: List[str],
    min_score: float,
) -> Dict[str, Any]:
    reuse_min_score = float(min_score)

    prepared_results: List[Dict[str, Any]] = []
    result_lookup: Dict[tuple, Dict[str, Any]] = {}
    for item in selected_results:
        cloned = dict(item)
        direct_queries = [str(query).strip() for query in cloned.get("matched_sub_queries", []) if str(query).strip()]
        cloned["direct_supported_sub_queries"] = direct_queries
        cloned["reused_supported_sub_queries"] = [
            str(query).strip() for query in cloned.get("reused_supported_sub_queries", []) if str(query).strip()
        ]
        prepared_results.append(cloned)
        result_lookup[_result_identity(cloned)] = cloned

    if not prepared_results or not unsupported_sub_queries:
        return {
            "results": prepared_results,
            "direct_unsupported_sub_queries": list(unsupported_sub_queries),
            "final_unsupported_sub_queries": list(unsupported_sub_queries),
            "reused_supported_sub_queries": [],
            "reuse_support_details": [],
            "reuse_min_score": reuse_min_score,
        }

    retrieval_query_lookup = {
        str(item.get("query") or "").strip(): str(item.get("retrieval_query") or item.get("query") or "").strip()
        for item in sub_query_support
        if str(item.get("query") or "").strip()
    }

    reused_supported_sub_queries: List[str] = []
    final_unsupported_sub_queries: List[str] = []
    reuse_support_details: List[Dict[str, Any]] = []

    for sub_query in unsupported_sub_queries:
        probe_query = retrieval_query_lookup.get(sub_query) or sub_query
        support_entries = vector_store_instance.probe_query_support_for_results(
            query_text=probe_query,
            results=prepared_results,
        )
        best_score = 0.0
        supported_pages: List[Dict[str, Any]] = []
        reuse_applied = False

        for entry in support_entries:
            score = float(entry.get("score", 0.0))
            best_score = max(best_score, score)
            if score < reuse_min_score:
                continue

            key = (
                entry.get("document_id", ""),
                entry.get("page_number", -1),
                entry.get("image_path", ""),
            )
            result = result_lookup.get(key)
            if result is None:
                continue

            reused_queries = result.setdefault("reused_supported_sub_queries", [])
            if sub_query not in reused_queries:
                reused_queries.append(sub_query)
            reuse_applied = True
            supported_pages.append(
                {
                    "page_number": int(entry.get("page_number") or 0),
                    "score": round(score, 2),
                }
            )

        if reuse_applied:
            reused_supported_sub_queries.append(sub_query)
        else:
            final_unsupported_sub_queries.append(sub_query)

        reuse_support_details.append(
            {
                "query": sub_query,
                "probe_query": probe_query,
                "best_selected_evidence_score": round(best_score, 2),
                "reuse_applied": reuse_applied,
                "supported_pages": supported_pages,
            }
        )

    return {
        "results": prepared_results,
        "direct_unsupported_sub_queries": list(unsupported_sub_queries),
        "final_unsupported_sub_queries": final_unsupported_sub_queries,
        "reused_supported_sub_queries": reused_supported_sub_queries,
        "reuse_support_details": reuse_support_details,
        "reuse_min_score": reuse_min_score,
    }


def _merge_result_groups(
    result_groups: List[List[Dict[str, Any]]],
    sub_queries: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    merged_by_key: Dict[tuple, Dict[str, Any]] = {}
    merged_order: List[Dict[str, Any]] = []
    max_group_len = max((len(group) for group in result_groups), default=0)

    for rank in range(max_group_len):
        for part_index, group in enumerate(result_groups):
            if rank >= len(group):
                continue

            candidate = dict(group[rank])
            key = _result_identity(candidate)
            if key in merged_by_key:
                existing = merged_by_key[key]
                existing["score"] = max(float(existing.get("score", 0.0)), float(candidate.get("score", 0.0)))
                matched = existing.setdefault("matched_sub_queries", [])
                sub_query = sub_queries[part_index]
                if sub_query not in matched:
                    matched.append(sub_query)
                continue

            candidate["matched_sub_queries"] = [sub_queries[part_index]]
            merged_by_key[key] = candidate
            merged_order.append(candidate)

    return merged_order[:limit]


def _aggregate_compound_retrieval_timing(
    sub_queries: List[str],
    retrieval_queries: List[str],
    timings: List[Dict[str, Any]],
    returned_points: int,
) -> Dict[str, Any]:
    return {
        "query_embedding_ms": round(sum(float(item.get("query_embedding_ms", 0.0)) for item in timings), 2),
        "qdrant_query_ms": round(sum(float(item.get("qdrant_query_ms", 0.0)) for item in timings), 2),
        "result_format_ms": round(sum(float(item.get("result_format_ms", 0.0)) for item in timings), 2),
        "total_retrieval_ms": round(sum(float(item.get("total_retrieval_ms", 0.0)) for item in timings), 2),
        "prefetch_limit": sum(int(item.get("prefetch_limit", 0)) for item in timings),
        "returned_points": returned_points,
        "sub_query_count": len(sub_queries),
        "sub_query_timings": [
            {
                "query": sub_query,
                "retrieval_query": retrieval_query,
                "query_embedding_ms": timing.get("query_embedding_ms", 0.0),
                "qdrant_query_ms": timing.get("qdrant_query_ms", 0.0),
                "result_format_ms": timing.get("result_format_ms", 0.0),
                "total_retrieval_ms": timing.get("total_retrieval_ms", 0.0),
                "returned_points": timing.get("returned_points", 0),
            }
            for sub_query, retrieval_query, timing in zip(sub_queries, retrieval_queries, timings)
        ],
    }


def _retrieve_compound_aware(
    vector_store_instance,
    query_text: str,
    document_ids: Optional[List[str]],
    top_k: int,
    min_score: float,
    allow_compound_split: bool = True,
    query_labels: Optional[List[str]] = None,
    retrieval_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    normalized_query_text = re.sub(r"\s+", " ", (query_text or "").strip())
    if query_labels and retrieval_queries and len(query_labels) == len(retrieval_queries):
        sub_queries = [re.sub(r"\s+", " ", str(item or "").strip()) for item in query_labels if str(item or "").strip()]
        normalized_retrieval_queries = [
            re.sub(r"\s+", " ", str(item or "").strip())
            for item in retrieval_queries[:len(sub_queries)]
            if str(item or "").strip()
        ]
        if len(normalized_retrieval_queries) != len(sub_queries):
            sub_queries = _split_compound_query(normalized_query_text) if allow_compound_split else ([normalized_query_text] if normalized_query_text else [])
            normalized_retrieval_queries = list(sub_queries)
    else:
        sub_queries = _split_compound_query(normalized_query_text) if allow_compound_split else ([normalized_query_text] if normalized_query_text else [])
        normalized_retrieval_queries = list(sub_queries)

    if len(sub_queries) <= 1:
        retrieval_query_text = normalized_retrieval_queries[0] if normalized_retrieval_queries else (normalized_query_text or query_text)
        retrieval_payload = vector_store_instance.retrieve_with_two_stage(
            query_text=retrieval_query_text,
            document_ids=document_ids,
            top_k=top_k,
        )
        results = retrieval_payload["results"]
        selection_meta = _select_results_with_threshold_fallback(
            results,
            min_score,
            fallback_limit=1,
            selection_limit=top_k,
        )
        selected_results = selection_meta["selected_results"]
        sub_query_text = sub_queries[0] if sub_queries else retrieval_query_text
        return {
            "sub_queries": sub_queries or [query_text],
            "retrieval_sub_queries": normalized_retrieval_queries or [retrieval_query_text],
            "results": results,
            "selected_results": selected_results,
            "unsupported_sub_queries": [] if selected_results else ([sub_query_text] if sub_query_text else []),
            "fallback_details": [
                {
                    "query": sub_query_text,
                    "retrieval_query": retrieval_query_text,
                    "fallback_used": bool(selection_meta.get("fallback_used")),
                    "fallback_count": int(selection_meta.get("fallback_count") or 0),
                    "fallback_best_score": float(selection_meta.get("fallback_best_score") or 0.0),
                }
            ] if sub_query_text else [],
            "sub_query_support": [
                {
                    "query": sub_query_text,
                    "retrieval_query": retrieval_query_text,
                    "best_score": _best_result_score(results),
                    "selected_count": len(selected_results),
                    "fallback_used": bool(selection_meta.get("fallback_used")),
                    "fallback_best_score": float(selection_meta.get("fallback_best_score") or 0.0),
                }
            ] if sub_query_text else [],
            "timing": retrieval_payload["timing"],
        }

    per_sub_query_top_k = max(2, min(3, top_k))
    selected_limit = min(8, max(top_k, len(sub_queries) * per_sub_query_top_k))
    candidate_limit = min(12, max(selected_limit, len(sub_queries) * per_sub_query_top_k))

    raw_result_groups: List[List[Dict[str, Any]]] = []
    selected_result_groups: List[List[Dict[str, Any]]] = []
    sub_query_timings: List[Dict[str, Any]] = []
    unsupported_sub_queries: List[str] = []
    sub_query_support: List[Dict[str, Any]] = []
    fallback_details: List[Dict[str, Any]] = []

    for sub_query, retrieval_query in zip(sub_queries, normalized_retrieval_queries):
        retrieval_payload = vector_store_instance.retrieve_with_two_stage(
            query_text=retrieval_query,
            document_ids=document_ids,
            top_k=per_sub_query_top_k,
        )
        raw_results = retrieval_payload["results"]
        selection_meta = _select_results_with_threshold_fallback(
            raw_results,
            min_score,
            fallback_limit=1,
            selection_limit=per_sub_query_top_k,
        )
        selected_results = selection_meta["selected_results"]
        raw_result_groups.append(raw_results)
        selected_result_groups.append(selected_results)
        sub_query_timings.append(retrieval_payload["timing"])
        if not selected_results:
            unsupported_sub_queries.append(sub_query)
        fallback_details.append(
            {
                "query": sub_query,
                "retrieval_query": retrieval_query,
                "fallback_used": bool(selection_meta.get("fallback_used")),
                "fallback_count": int(selection_meta.get("fallback_count") or 0),
                "fallback_best_score": float(selection_meta.get("fallback_best_score") or 0.0),
            }
        )
        sub_query_support.append(
            {
                "query": sub_query,
                "retrieval_query": retrieval_query,
                "best_score": _best_result_score(raw_results),
                "selected_count": len(selected_results),
                "fallback_used": bool(selection_meta.get("fallback_used")),
                "fallback_best_score": float(selection_meta.get("fallback_best_score") or 0.0),
            }
        )

    merged_results = _merge_result_groups(raw_result_groups, sub_queries, candidate_limit)
    merged_selected_results = _merge_result_groups(selected_result_groups, sub_queries, selected_limit)

    return {
        "sub_queries": sub_queries,
        "retrieval_sub_queries": normalized_retrieval_queries,
        "results": merged_results,
        "selected_results": merged_selected_results,
        "unsupported_sub_queries": unsupported_sub_queries,
        "fallback_details": fallback_details,
        "sub_query_support": sub_query_support,
        "timing": _aggregate_compound_retrieval_timing(sub_queries, normalized_retrieval_queries, sub_query_timings, len(merged_results)),
    }


def _build_confidence_summary(results: List[dict]) -> dict:
    scores = sorted((float(item.get("score", 0.0)) for item in results), reverse=True)[:3]
    if not scores:
        return {"label": "低", "score": 0.0, "sample_size": 0, "tier": "low"}

    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.85:
        label = "高"
        tier = "high"
    elif avg_score >= 0.7:
        label = "中"
        tier = "medium"
    else:
        label = "低"
        tier = "low"

    return {
        "label": label,
        "score": round(avg_score, 2),
        "sample_size": len(scores),
        "tier": tier,
    }


def _rebuild_image_cache_if_needed(results: list) -> None:
    """
    检查检索结果中的图像缓存文件是否存在（系统重启后 /tmp 会被清空）。
    若缺失，则从持久化原件目录中找到对应文件并重新渲染图像缓存。
    """
    missing_doc_ids: dict = {}
    for r in results:
        image_path = str(r.get("image_path", ""))
        if image_path and not os.path.exists(image_path):
            doc_id = r.get("document_id", "")
            if doc_id and doc_id not in missing_doc_ids:
                missing_doc_ids[doc_id] = r.get("document_name", doc_id)

    if not missing_doc_ids:
        return

    # 目录名继续沿用 pdfs 以兼容已有持久化数据，实际存放的是所有原始上传文件。
    stored_files_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
    for doc_id, doc_name in missing_doc_ids.items():
        matches = glob.glob(os.path.join(stored_files_dir, f"{doc_id}_*"))
        file_path = matches[0] if matches else None
        if not file_path or not os.path.exists(file_path):
            print(f"[Warning] File not found for document '{doc_name}' ({doc_id}), cannot rebuild image cache.")
            continue
        print(f"[Info] Rebuilding image cache for: {doc_name}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                process_pdf_to_images(file_path)
            elif ext == ".pptx":
                process_pptx_to_images(file_path)
            elif ext in {".txt", ".md"}:
                process_text_to_images(file_path)
            else:
                process_image_to_images(file_path)
        except Exception as rebuild_err:
            print(f"[Warning] Failed to rebuild cache for {doc_name}: {rebuild_err}")


@router.get("/files/{document_id}/download")
def download_file(document_id: str):
    """
    Download the original uploaded file.
    """
    stored_files_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
    matches = glob.glob(os.path.join(stored_files_dir, f"{document_id}_*"))
    
    file_path = matches[0] if matches else None
        
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    mime = mime_map.get(ext, "application/octet-stream")
    
    return FileResponse(
        file_path, 
        media_type=mime, 
        content_disposition_type="inline", 
        filename=filename
    )

@router.get("/files")
def list_files():
    """
    Returns a list of all uniquely uploaded documents in the vector store.
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant backend is not ready")

    try:
        files = vector_store_instance.get_all_documents()
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{document_id}")
def delete_file(document_id: str):
    """
    Deletes a document from the vector store by its ID.
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant backend is not ready")
        
    try:
        vector_store_instance.delete_document(document_id)
        
        stored_files_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
        for m in glob.glob(os.path.join(stored_files_dir, f"{document_id}_*")):
            os.remove(m)
            
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".pptx"}

_UPLOAD_STAGE_RANGES = {
    "queued": (15.0, 15.0),
    "rendering": (15.0, 35.0),
    "embedding": (35.0, 85.0),
    "upserting": (85.0, 98.0),
    "done": (100.0, 100.0),
}


def _get_upload_job_manager_or_raise():
    from backend.main import upload_job_manager

    if upload_job_manager is None:
        raise RuntimeError("上传任务管理器尚未就绪")
    return upload_job_manager


def _get_vector_store_or_raise():
    from backend.main import vector_store_instance

    if vector_store_instance is None:
        raise RuntimeError("Qdrant 后端尚未就绪")
    return vector_store_instance


def _compute_stage_progress(stage: str, current: int = 0, total: int = 0) -> float:
    if stage == "done":
        return 100.0

    start, end = _UPLOAD_STAGE_RANGES.get(stage, (0.0, 0.0))
    if total <= 0 or end <= start:
        return round(end, 1)

    ratio = max(0.0, min(float(current) / float(total), 1.0))
    return round(start + (end - start) * ratio, 1)


def _build_upload_stage_message(stage: str, suffix: str) -> str:
    if stage == "queued":
        return "文件已上传，等待开始处理..."
    if stage == "rendering":
        if suffix == ".pdf":
            return "正在将 PDF 转为页面图..."
        if suffix == ".pptx":
            return "正在将 PPTX 转为页面图..."
        return "正在整理图片页面..."
    if stage == "embedding":
        return "正在建立 ColPali 视觉索引..."
    if stage == "upserting":
        return "正在写入向量库..."
    if stage == "done":
        return "文档已完成解析并建立索引。"
    if stage == "error":
        return "文档处理失败。"
    return "处理中..."


def _update_upload_job(
    task_id: str,
    *,
    stage: str,
    suffix: str,
    status: Optional[str] = None,
    current: Optional[int] = None,
    total: Optional[int] = None,
    progress_percent: Optional[float] = None,
    message: Optional[str] = None,
    **fields,
) -> Optional[Dict[str, Any]]:
    manager = _get_upload_job_manager_or_raise()
    previous_job = manager.get_job(task_id) or {}
    resolved_current = int(current if current is not None else previous_job.get("stage_current", 0) or 0)
    resolved_total = int(total if total is not None else previous_job.get("stage_total", 0) or 0)

    if progress_percent is None:
        if stage == "error":
            progress_percent = float(previous_job.get("progress_percent", 0.0) or 0.0)
        else:
            progress_percent = _compute_stage_progress(stage, resolved_current, resolved_total)

    payload = {
        "status": status or ("done" if stage == "done" else "error" if stage == "error" else "running"),
        "stage": stage,
        "message": message or _build_upload_stage_message(stage, suffix),
        "progress_percent": float(progress_percent),
        "stage_current": resolved_current,
        "stage_total": resolved_total,
    }
    payload.update(fields)
    return manager.update_job(task_id, **payload)


def _run_upload_job(task_id: str, temp_path: str, original_filename: str, suffix: str) -> None:
    final_file_path = None
    image_paths: List[str] = []
    document_id = ""

    try:
        vector_store_instance = _get_vector_store_or_raise()
        document_id = get_file_hash(temp_path)
        _update_upload_job(
            task_id,
            stage="queued",
            status="queued",
            suffix=suffix,
            progress_percent=15.0,
        )

        render_start = time.perf_counter()

        def on_render_progress(current: int, total: int) -> None:
            _update_upload_job(
                task_id,
                stage="rendering",
                suffix=suffix,
                current=current,
                total=total,
                page_count=max(int(total), 0),
            )

        _update_upload_job(task_id, stage="rendering", suffix=suffix, current=0, total=1)
        if suffix == ".pdf":
            image_paths = process_pdf_to_images(temp_path, on_progress=on_render_progress)
        elif suffix == ".pptx":
            image_paths = process_pptx_to_images(temp_path, on_progress=on_render_progress)
        else:
            image_paths = process_image_to_images(temp_path, on_progress=on_render_progress)

        document_render_ms = (time.perf_counter() - render_start) * 1000
        if not image_paths:
            raise RuntimeError("未能成功解析出任何页面图像")

        safe_filename = (original_filename or os.path.basename(temp_path) or "uploaded_file")
        safe_filename = safe_filename.replace("/", "_").replace("\\", "_").replace(" ", "_")
        stored_files_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
        os.makedirs(stored_files_dir, exist_ok=True)
        final_file_path = os.path.join(stored_files_dir, f"{document_id}_{safe_filename}")
        shutil.copy(temp_path, final_file_path)

        _update_upload_job(
            task_id,
            stage="embedding",
            suffix=suffix,
            current=0,
            total=max(len(image_paths), 1),
            page_count=len(image_paths),
            timing={"document_render_ms": round(document_render_ms, 2)},
        )

        def on_index_progress(event: Dict[str, Any]) -> None:
            stage = str(event.get("stage") or "embedding")
            completed = int(event.get("completed") or 0)
            total = int(event.get("total") or len(image_paths) or 1)
            _update_upload_job(
                task_id,
                stage=stage,
                suffix=suffix,
                current=completed,
                total=total,
                page_count=len(image_paths),
            )

        index_result = vector_store_instance.embed_and_store_documents(
            image_paths=image_paths,
            document_id=document_id,
            document_name=original_filename,
            on_progress=on_index_progress,
        )
        if not index_result.get("ok"):
            raise RuntimeError("存储多模态特征库发生意外错误")

        upload_timing = {
            "document_render_ms": round(document_render_ms, 2),
            **index_result.get("timing", {}),
        }
        logger.info("upload_timing file=%s timing=%s", original_filename, upload_timing)
        print(
            "[Upload] "
            f"file={original_filename} render={upload_timing['document_render_ms']}ms "
            f"embed={upload_timing.get('embedding_ms', 0)}ms build={upload_timing.get('point_build_ms', 0)}ms "
            f"upsert={upload_timing.get('qdrant_upsert_ms', 0)}ms total_index={upload_timing.get('total_index_ms', 0)}ms"
        )

        result_payload = {
            "status": "success",
            "document_id": document_id,
            "document_name": original_filename,
            "page_count": len(image_paths),
            "timing": upload_timing,
        }
        _update_upload_job(
            task_id,
            stage="done",
            status="done",
            suffix=suffix,
            progress_percent=100.0,
            page_count=len(image_paths),
            timing=upload_timing,
            result=result_payload,
            error=None,
        )
    except Exception as exc:
        logger.exception("upload_job_failed task_id=%s filename=%s", task_id, original_filename)
        error_message = str(exc).strip() or "未知错误"
        try:
            vector_store_instance = _get_vector_store_or_raise()
            if document_id:
                vector_store_instance.delete_document(document_id)
        except Exception:
            logger.exception("upload_job_cleanup_failed task_id=%s document_id=%s", task_id, document_id)

        if final_file_path and os.path.exists(final_file_path):
            try:
                os.remove(final_file_path)
            except OSError:
                logger.warning("failed_to_remove_failed_upload_file path=%s", final_file_path)

        _update_upload_job(
            task_id,
            stage="error",
            status="error",
            suffix=suffix,
            message=f"文档处理失败：{error_message}",
            error=error_message,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/jobs/{task_id}")
def get_upload_job(task_id: str):
    try:
        job = _get_upload_job_manager_or_raise().get_job(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if job is None:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return {"status": "success", "job": job}


@router.get("/jobs/{task_id}/events")
async def stream_upload_job_events(task_id: str, request: Request):
    try:
        manager = _get_upload_job_manager_or_raise()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    initial_job = manager.get_job(task_id)
    if initial_job is None:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")

    async def event_stream():
        last_revision = -1
        while True:
            if await request.is_disconnected():
                break

            job = manager.get_job(task_id)
            if job is None:
                payload = {"type": "error", "data": {"message": "上传任务不存在或已过期"}}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            revision = int(job.get("revision", 0) or 0)
            if revision != last_revision:
                event_type = "progress"
                if job.get("status") == "done":
                    event_type = "done"
                elif job.get("status") == "error":
                    event_type = "error"

                payload = {"type": event_type, "data": job}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last_revision = revision

                if event_type in {"done", "error"}:
                    break

            await asyncio.sleep(0.35)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/upload")
def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    接收视觉文档文件，处理为页面缓存与向量特征。
    支持格式：PDF / PNG / JPG / JPEG / WEBP / PPTX
    """
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="请上传支持的文件格式（PDF / PNG / JPG / WEBP / PPTX）")
    tmp = tempfile.NamedTemporaryFile(prefix="rag_upload_", suffix=suffix, delete=False)
    temp_path = tmp.name
    tmp.close()
    job_scheduled = False
    try:
        # 将流保存到本地临时文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_hash = get_file_hash(temp_path)
        job = _get_upload_job_manager_or_raise().create_job(
            filename=file.filename or os.path.basename(temp_path),
            document_id=file_hash,
        )
        background_tasks.add_task(
            _run_upload_job,
            job["task_id"],
            temp_path,
            file.filename or os.path.basename(temp_path),
            suffix,
        )
        job_scheduled = True
        return {
            "status": "accepted",
            "task_id": job["task_id"],
            "document_id": file_hash,
            "document_name": file.filename,
            "job": job,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 已交给后台任务后，临时文件由任务自己清理。
        if not job_scheduled and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/files/{document_id}/suggestions")
def get_document_suggestions(document_id: str, max_questions: int = 4):
    from backend.main import vector_store_instance

    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

    sample_pages = vector_store_instance.get_document_page_samples(
        document_id=document_id,
        limit=max(3, min(max_questions, 5)),
    )
    if not sample_pages:
        raise HTTPException(status_code=404, detail="未找到可用于生成建议问题的文档页面")

    _rebuild_image_cache_if_needed(sample_pages)
    available_pages = [page for page in sample_pages if os.path.exists(str(page.get("image_path", "")))]
    if not available_pages:
        raise HTTPException(status_code=404, detail="建议问题所需的页面缓存不可用")

    evidence_context = []
    image_paths = []
    for index, page in enumerate(available_pages, start=1):
        image_paths.append(page["image_path"])
        evidence_context.append(
            {
                "evidence_id": f"E{index}",
                "document_name": page.get("document_name", "Unknown File"),
                "page_number": page.get("page_number", 0),
            }
        )

    questions = generate_suggested_questions(
        document_name=available_pages[0].get("document_name", document_id),
        image_paths=image_paths,
        evidence_context=evidence_context,
        max_questions=max_questions,
    )

    return {
        "status": "success",
        "document_id": document_id,
        "document_name": available_pages[0].get("document_name", document_id),
        "questions": questions,
    }

@router.post("/chat")
async def chat(request: Request, req: ChatRequest):
    """
    接收对某几篇（或全部）文档的查询，通过 SSE 流式返回：
    第一个事件为 evidence 卡片数据，后续事件为逐 token 文字，最终发送 [DONE]。
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

    pre_guard = _guard_obvious_out_of_scope(req.query)

    results = []
    score_filtered: List[Dict[str, Any]] = []
    retrieval_timing = None
    retrieval_query = req.query
    query_plan = _plan_retrieval_queries(req.query, req.chat_history)
    original_sub_queries = list(query_plan.get("query_labels") or [req.query])
    retrieval_sub_queries = list(query_plan.get("retrieval_sub_queries") or [req.query])
    compound_split_allowed = len(original_sub_queries) > 1
    sub_queries = original_sub_queries or [req.query]
    unsupported_sub_queries: List[str] = []
    fallback_details: List[Dict[str, Any]] = []
    rewrite_meta: Dict[str, Any] = dict(query_plan.get("rewrite_meta") or {
        "applied": False,
        "rewritten_query": req.query,
        "reason": "not_attempted",
        "timing_ms": 0.0,
        "sub_query_rewrites": [],
    })
    if len(retrieval_sub_queries) == 1:
        retrieval_query = retrieval_sub_queries[0]

    # ── 同步完成 RAG 检索（流式响应开始前必须就绪）──
    if not pre_guard:
        retrieval_payload = _retrieve_compound_aware(
            vector_store_instance=vector_store_instance,
            query_text=retrieval_query,
            document_ids=req.document_ids,
            top_k=req.top_k,
            min_score=req.min_score,
            allow_compound_split=compound_split_allowed,
            query_labels=original_sub_queries,
            retrieval_queries=retrieval_sub_queries,
        )
        results = retrieval_payload["results"]
        score_filtered = retrieval_payload["selected_results"]
        unsupported_sub_queries = list(retrieval_payload.get("unsupported_sub_queries") or [])
        fallback_details = list(retrieval_payload.get("fallback_details") or [])
        retrieval_timing = retrieval_payload["timing"]
        fallback_sub_queries = [
            str(item.get("query") or "").strip()
            for item in fallback_details
            if bool(item.get("fallback_used")) and str(item.get("query") or "").strip()
        ]
        fallback_best_score = max(
            (float(item.get("fallback_best_score") or 0.0) for item in fallback_details if bool(item.get("fallback_used"))),
            default=0.0,
        )
        retrieval_timing.update(
            {
                "rewrite_applied": bool(rewrite_meta.get("applied")),
                "rewrite_ms": float(rewrite_meta.get("timing_ms") or 0.0),
                "rewrite_reason": str(rewrite_meta.get("reason") or "unknown"),
                "rewrite_history_messages": int(rewrite_meta.get("history_messages_used") or 0),
                "rewrite_model": str(rewrite_meta.get("model") or ""),
                "retrieval_query": retrieval_query,
                "retrieval_sub_queries": retrieval_sub_queries,
                "sub_query_rewrites": list(rewrite_meta.get("sub_query_rewrites") or []),
                "original_sub_query_count": len(original_sub_queries) or 1,
                "compound_split_applied": compound_split_allowed,
                "compound_split_source": "original_query",
                "min_score_applied": float(req.min_score),
                "fallback_evidence_used": bool(fallback_sub_queries),
                "fallback_evidence_count": len(fallback_sub_queries),
                "fallback_best_score": round(fallback_best_score, 2),
                "fallback_sub_queries": fallback_sub_queries,
                "fallback_details": fallback_details,
                "unsupported_sub_query_count": len(unsupported_sub_queries),
                "unsupported_sub_queries": unsupported_sub_queries,
                "sub_query_support": list(retrieval_payload.get("sub_query_support") or []),
            }
        )
        sub_queries = retrieval_payload["sub_queries"]

    _rebuild_image_cache_if_needed(results)
    valid_results = [r for r in score_filtered if os.path.exists(str(r.get("image_path", "")))]
    reuse_support_meta = _apply_reusable_sub_query_support(
        vector_store_instance=vector_store_instance,
        selected_results=valid_results,
        sub_query_support=list(retrieval_timing.get("sub_query_support") or []) if retrieval_timing else [],
        unsupported_sub_queries=unsupported_sub_queries,
        min_score=req.min_score,
    ) if (not pre_guard and valid_results) else {
        "results": [
            {
                **dict(item),
                "direct_supported_sub_queries": list(item.get("matched_sub_queries", [])),
                "reused_supported_sub_queries": [],
            }
            for item in valid_results
        ],
        "direct_unsupported_sub_queries": list(unsupported_sub_queries),
        "final_unsupported_sub_queries": list(unsupported_sub_queries),
        "reused_supported_sub_queries": [],
        "reuse_support_details": [],
        "reuse_min_score": float(req.min_score),
    }
    valid_results = reuse_support_meta["results"]
    direct_unsupported_sub_queries = list(reuse_support_meta.get("direct_unsupported_sub_queries") or [])
    unsupported_sub_queries = list(reuse_support_meta.get("final_unsupported_sub_queries") or [])
    reused_supported_sub_queries = list(reuse_support_meta.get("reused_supported_sub_queries") or [])
    evidence_images = [r["image_path"] for r in valid_results if r.get("image_path")]
    confidence = _build_confidence_summary(valid_results)
    if retrieval_timing is not None:
        retrieval_timing.update(
            {
                "direct_unsupported_sub_queries": direct_unsupported_sub_queries,
                "unsupported_sub_queries": unsupported_sub_queries,
                "unsupported_sub_query_count": len(unsupported_sub_queries),
                "reused_supported_sub_queries": reused_supported_sub_queries,
                "reused_supported_sub_query_count": len(reused_supported_sub_queries),
                "reuse_probe_min_score": float(reuse_support_meta.get("reuse_min_score") or req.min_score),
                "reuse_support_details": list(reuse_support_meta.get("reuse_support_details") or []),
            }
        )

    used_keys = {(r.get("document_id", ""), r.get("page_number", -1)) for r in valid_results}
    evidence_id_map = {}
    frontend_evidences = []
    evidence_context = []
    for index, r in enumerate(valid_results, start=1):
        image_path = str(r.get("image_path", ""))
        if not image_path:
            continue
        key = (r.get("document_id", ""), r.get("page_number", -1))
        evidence_id = f"E{index}"
        evidence_id_map[key] = evidence_id
        direct_supported_sub_queries = list(r.get("direct_supported_sub_queries", []))
        reused_supported_sub_queries = list(r.get("reused_supported_sub_queries", []))
        all_supported_sub_queries = direct_supported_sub_queries + [
            query for query in reused_supported_sub_queries if query not in direct_supported_sub_queries
        ]
        evidence_context.append({
            "evidence_id": evidence_id,
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "fallback_below_threshold": bool(r.get("fallback_below_threshold")),
            "fallback_source_score": float(r.get("fallback_source_score", r.get("score", 0.0))),
            "fallback_gap_to_threshold": float(r.get("fallback_gap_to_threshold", 0.0)),
            "fallback_tier": str(r.get("fallback_tier") or "formal"),
            "fallback_reason": str(r.get("fallback_reason") or ""),
            "matched_sub_queries": direct_supported_sub_queries,
            "direct_supported_sub_queries": direct_supported_sub_queries,
            "reused_supported_sub_queries": reused_supported_sub_queries,
            "all_supported_sub_queries": all_supported_sub_queries,
        })
        frontend_evidences.append({
            "evidence_id": evidence_id,
            "document_id": r.get("document_id", ""),
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "fallback_below_threshold": bool(r.get("fallback_below_threshold")),
            "fallback_source_score": float(r.get("fallback_source_score", r.get("score", 0.0))),
            "fallback_gap_to_threshold": float(r.get("fallback_gap_to_threshold", 0.0)),
            "fallback_tier": str(r.get("fallback_tier") or "formal"),
            "fallback_reason": str(r.get("fallback_reason") or ""),
            "image_base64": image_path_to_base64(image_path),
            "matched_sub_queries": direct_supported_sub_queries,
            "reused_supported_sub_queries": reused_supported_sub_queries,
        })

    # 全部候选页（top_k 个，含未被 min_score 采用的），供前端展示所有得分
    all_candidates_fe = []
    for r in results:
        image_path = str(r.get("image_path", ""))
        if not image_path:
            continue
        key = (r.get("document_id", ""), r.get("page_number", -1))
        score = float(r.get("score", 0.0))
        unused_reason = None
        if key not in used_keys:
            unused_reason = f"低于阈值 {req.min_score:.2f} 未采用" if score < req.min_score else "未进入最终正式依据"
        all_candidates_fe.append({
            "evidence_id": evidence_id_map.get(key),
            "document_id": r.get("document_id", ""),
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": score,
            "image_base64": image_path_to_base64(image_path),
            "matched_sub_queries": list(r.get("matched_sub_queries", [])),
            "is_used": key in used_keys,
            "unused_reason": unused_reason,
        })

    if retrieval_timing is not None:
        retrieval_timing.update(
            {
                "selected_evidence_count": len(frontend_evidences),
                "unused_candidate_count": sum(1 for item in all_candidates_fe if not item.get("is_used")),
            }
        )

    if pre_guard:
        guard_payload = pre_guard
        logger.info(
            "chat_guarded stage=%s reason=%s query=%r document_scope=%s meta=%s",
            guard_payload["stage"],
            guard_payload["reason"],
            req.query,
            req.document_ids,
            guard_payload.get("meta", {}),
        )
        print(
            f"[Chat Guard] stage={guard_payload['stage']} reason={guard_payload['reason']} "
            f"document_scope={req.document_ids} meta={guard_payload.get('meta', {})}"
        )
    else:
        logger.info(
            "chat_retrieval timing=%s confidence=%s document_scope=%s sub_queries=%s rewrite=%s",
            retrieval_timing,
            confidence,
            req.document_ids,
            sub_queries,
            rewrite_meta,
        )
        print(
            f"[Chat] confidence={confidence} timing={retrieval_timing} "
            f"rewrite_applied={rewrite_meta.get('applied')} retrieval_query={retrieval_query!r} "
            f"document_scope={req.document_ids} sub_queries={sub_queries}"
        )

    async def event_stream():
        if pre_guard:
            guard_payload = pre_guard
            yield f"data: {json.dumps({'type': 'guarded', 'data': guard_payload})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 事件 1: 证据卡片 ──
        if await request.is_disconnected():
            return

        yield f"data: {json.dumps({'type': 'evidences', 'data': {'evidences': frontend_evidences, 'all_candidates': all_candidates_fe, 'confidence': confidence, 'retrieval_timing': retrieval_timing}})}\n\n"

        if not evidence_images:
            no_evidence_answer = _build_insufficient_evidence_answer(req.min_score, sub_queries)
            yield f"data: {json.dumps({'type': 'token', 'data': no_evidence_answer})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 事件 2..N: 逐 token 文字 ──
        token_stream = generate_answer_stream(
            query_text=req.query,
            image_paths=evidence_images,
            chat_history=req.chat_history,
            evidence_context=evidence_context,
            sub_queries=sub_queries,
            unsupported_sub_queries=unsupported_sub_queries,
        )
        generation_start = time.perf_counter()
        first_token_ms = None
        token_count = 0
        try:
            for token in token_stream:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - generation_start) * 1000
                token_count += 1
                if await request.is_disconnected():
                    return
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        finally:
            total_generation_ms = (time.perf_counter() - generation_start) * 1000
            generation_timing = {
                "first_token_ms": round(first_token_ms or 0.0, 2),
                "total_generation_ms": round(total_generation_ms, 2),
                "token_count": token_count,
            }
            logger.info("chat_generation timing=%s document_scope=%s", generation_timing, req.document_ids)
            print(
                "[Generation] "
                f"first_token={generation_timing['first_token_ms']}ms "
                f"total={generation_timing['total_generation_ms']}ms tokens={token_count} "
                f"document_scope={req.document_ids}"
            )
            token_stream.close()

        # ── 结束哨兵 ──
        if await request.is_disconnected():
            return
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

def image_path_to_base64(image_path: str) -> str:
    import base64
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    mime = "image/png"
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"