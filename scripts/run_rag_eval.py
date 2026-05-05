#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import requests
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("requests is required. Install dependencies from requirements.txt first.") from exc

from src.eval.faithfulness_judge import score_multimodal_faithfulness


DEFAULT_EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", str(REPO_ROOT / "data" / "eval_runs")))


def ensure_default_eval_runs_dir() -> Path:
    legacy_eval_runs_dir = REPO_ROOT / "eval_runs"
    if not DEFAULT_EVAL_RUNS_DIR.exists() and legacy_eval_runs_dir.exists() and legacy_eval_runs_dir != DEFAULT_EVAL_RUNS_DIR:
        DEFAULT_EVAL_RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
        legacy_eval_runs_dir.rename(DEFAULT_EVAL_RUNS_DIR)

    DEFAULT_EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_EVAL_RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small RAG benchmark against the live /api/rag/chat endpoint or summarize a reviewed CSV.",
    )
    parser.add_argument("--benchmark-file", help="Path to the benchmark JSON file.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--output-dir", help="Directory for run outputs. Defaults to data/eval_runs/<timestamp>.")
    parser.add_argument("--top-k", type=int, help="Override benchmark default top_k.")
    parser.add_argument("--min-score", type=float, help="Override benchmark default min_score.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Read timeout for each chat request.")
    parser.add_argument("--limit", type=int, help="Only run the first N benchmark items.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["two_stage", "colpali_only", "muvera_only"],
        default="two_stage",
        help="Retrieval mode to send to the backend during live eval runs.",
    )
    parser.add_argument(
        "--disable-automatic-faithfulness",
        action="store_true",
        help="Skip the multimodal faithfulness judge during live eval runs.",
    )
    parser.add_argument(
        "--faithfulness-model-name",
        help="Optional override for the model used by the multimodal faithfulness judge.",
    )
    parser.add_argument("--review-csv", help="If set, skip live requests and summarize a previously reviewed CSV.")
    return parser.parse_args()


def load_benchmark(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Benchmark file must contain a non-empty 'items' list.")

    for index, item in enumerate(items, start=1):
        validate_benchmark_item(item, index)
    return payload


def validate_benchmark_item(item: Dict[str, Any], index: int) -> None:
    item_id = str(item.get("id") or "").strip()
    question = str(item.get("question") or "").strip()
    gold_evidence = item.get("gold_evidence")
    if not item_id:
        raise ValueError(f"Benchmark item #{index} is missing 'id'.")
    if not question:
        raise ValueError(f"Benchmark item #{index} is missing 'question'.")
    if not isinstance(gold_evidence, list) or not gold_evidence:
        raise ValueError(f"Benchmark item {item_id!r} must define a non-empty 'gold_evidence' list.")

    for evidence_index, gold in enumerate(gold_evidence, start=1):
        if not isinstance(gold, dict):
            raise ValueError(f"Benchmark item {item_id!r} gold_evidence[{evidence_index}] must be an object.")
        if int(gold.get("page_number") or 0) <= 0:
            raise ValueError(f"Benchmark item {item_id!r} gold_evidence[{evidence_index}] needs a positive page_number.")
        document_id = str(gold.get("document_id") or "").strip()
        document_name = str(gold.get("document_name") or "").strip()
        if not document_id and not document_name:
            raise ValueError(
                f"Benchmark item {item_id!r} gold_evidence[{evidence_index}] must include document_id or document_name."
            )


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", lowered)
    return lowered


def extract_citations(answer: str) -> List[str]:
    return re.findall(r"\[(E\d+)\]", answer or "")


def build_output_dir(explicit_output_dir: Optional[str]) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_default_eval_runs_dir() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def iter_sse_events(response: requests.Response) -> Iterable[Dict[str, Any]]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[6:].strip()
        if payload == "[DONE]":
            yield {"type": "done", "data": None}
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def run_chat_sample(
    *,
    api_base_url: str,
    question: str,
    document_ids: Optional[List[str]],
    top_k: int,
    min_score: float,
    retrieval_mode: str,
    chat_history: Sequence[Dict[str, Any]],
    timeout_seconds: float,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    response = requests.post(
        f"{api_base_url.rstrip('/')}/api/rag/chat",
        json={
            "query": question,
            "document_ids": document_ids or None,
            "chat_history": list(chat_history or []),
            "top_k": top_k,
            "min_score": min_score,
            "retrieval_mode": retrieval_mode,
        },
        stream=True,
        timeout=(10, timeout_seconds),
    )
    response.raise_for_status()

    result: Dict[str, Any] = {
        "guarded": None,
        "evidences": [],
        "all_candidates": [],
        "confidence": None,
        "retrieval_timing": {},
        "answer_text": "",
        "client_first_token_ms": None,
        "client_total_chat_ms": None,
    }

    for event in iter_sse_events(response):
        event_type = event.get("type")
        if event_type == "guarded":
            result["guarded"] = event.get("data")
            continue
        if event_type == "error":
            raise RuntimeError(str(event.get("data") or "Unknown SSE error"))
        if event_type == "evidences":
            data = event.get("data") or {}
            result["evidences"] = list(data.get("evidences") or [])
            result["all_candidates"] = list(data.get("all_candidates") or [])
            result["confidence"] = data.get("confidence")
            result["retrieval_timing"] = dict(data.get("retrieval_timing") or {})
            continue
        if event_type == "token":
            if result["client_first_token_ms"] is None:
                result["client_first_token_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            result["answer_text"] += str(event.get("data") or "")
            continue
        if event_type == "done":
            break

    result["client_total_chat_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    return result


def evidence_matches_gold(prediction: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    predicted_page = int(prediction.get("page_number") or 0)
    gold_page = int(gold.get("page_number") or 0)
    if predicted_page != gold_page:
        return False

    gold_document_id = str(gold.get("document_id") or "").strip()
    predicted_document_id = str(prediction.get("document_id") or "").strip()
    if gold_document_id:
        if not predicted_document_id:
            return False
        if predicted_document_id != gold_document_id:
            return False

    gold_document_name = normalize_text(str(gold.get("document_name") or ""))
    predicted_document_name = normalize_text(str(prediction.get("document_name") or ""))
    if gold_document_name:
        return predicted_document_name == gold_document_name

    return True


def document_matches_gold(prediction: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    gold_document_id = str(gold.get("document_id") or "").strip()
    predicted_document_id = str(prediction.get("document_id") or "").strip()
    if gold_document_id:
        return predicted_document_id == gold_document_id

    gold_document_name = normalize_text(str(gold.get("document_name") or ""))
    predicted_document_name = normalize_text(str(prediction.get("document_name") or ""))
    return bool(gold_document_name) and predicted_document_name == gold_document_name


def hit_at_k(predictions: Sequence[Dict[str, Any]], gold_evidence: Sequence[Dict[str, Any]], k: int, *, doc_only: bool = False) -> int:
    truncated = predictions[:k]
    matcher = document_matches_gold if doc_only else evidence_matches_gold
    return int(any(matcher(prediction, gold) for prediction in truncated for gold in gold_evidence))


def reciprocal_rank(predictions: Sequence[Dict[str, Any]], gold_evidence: Sequence[Dict[str, Any]], *, doc_only: bool = False) -> float:
    matcher = document_matches_gold if doc_only else evidence_matches_gold
    for rank, prediction in enumerate(predictions, start=1):
        if any(matcher(prediction, gold) for gold in gold_evidence):
            return round(1.0 / rank, 4)
    return 0.0


def first_matching_rank(predictions: Sequence[Dict[str, Any]], gold_evidence: Sequence[Dict[str, Any]], *, doc_only: bool = False) -> Optional[int]:
    matcher = document_matches_gold if doc_only else evidence_matches_gold
    for rank, prediction in enumerate(predictions, start=1):
        if any(matcher(prediction, gold) for gold in gold_evidence):
            return rank
    return None


def compute_keyword_metrics(answer_text: str, keywords: Sequence[str]) -> Dict[str, Any]:
    normalized_answer = normalize_text(answer_text)
    cleaned_keywords = [keyword for keyword in (str(item).strip() for item in keywords) if keyword]
    if not cleaned_keywords:
        return {
            "keyword_coverage": None,
            "keyword_full_match": None,
            "matched_keywords": [],
        }

    matched_keywords = [keyword for keyword in cleaned_keywords if normalize_text(keyword) in normalized_answer]
    coverage = len(matched_keywords) / len(cleaned_keywords)
    return {
        "keyword_coverage": round(coverage, 4),
        "keyword_full_match": int(len(matched_keywords) == len(cleaned_keywords)),
        "matched_keywords": matched_keywords,
    }


def compute_answer_string_match(answer_text: str, gold_answer: str) -> Optional[int]:
    normalized_answer = normalize_text(answer_text)
    normalized_gold = normalize_text(gold_answer)
    if not normalized_gold:
        return None
    return int(normalized_gold in normalized_answer)


def format_evidence_list(evidences: Sequence[Dict[str, Any]], limit: int = 3) -> str:
    formatted = []
    for evidence in evidences[:limit]:
        doc_name = str(evidence.get("document_name") or evidence.get("document_id") or "unknown")
        page_number = int(evidence.get("page_number") or 0)
        score = float(evidence.get("score") or 0.0)
        formatted.append(f"{doc_name}#p{page_number}@{score:.2f}")
    return " | ".join(formatted)


def evaluate_sample(
    sample: Dict[str, Any],
    response_data: Dict[str, Any],
    *,
    enable_automatic_faithfulness: bool = True,
    faithfulness_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    gold_evidence = list(sample.get("gold_evidence") or [])
    raw_predictions = list(response_data.get("all_candidates") or [])
    final_predictions = list(response_data.get("evidences") or [])
    answer_text = str(response_data.get("answer_text") or "")
    retrieval_timing = dict(response_data.get("retrieval_timing") or {})
    citations = extract_citations(answer_text)
    evidence_lookup = {str(item.get("evidence_id") or ""): item for item in final_predictions}
    cited_gold_evidence = int(
        any(
            any(evidence_matches_gold(evidence_lookup[citation], gold) for gold in gold_evidence)
            for citation in citations
            if citation in evidence_lookup
        )
    )

    keyword_metrics = compute_keyword_metrics(answer_text, sample.get("gold_answer_keywords") or [])
    automatic_faithfulness = score_multimodal_faithfulness(
        question=str(sample.get("question") or ""),
        answer_text=answer_text,
        evidences=final_predictions,
        model_name=faithfulness_model_name,
    ) if enable_automatic_faithfulness else {
        "score": None,
        "label": "disabled",
        "reason": "Disabled by CLI flag.",
        "provider": "doubao_multimodal_judge",
    }

    return {
        "benchmark_id": str(sample.get("id") or ""),
        "question_type": str(sample.get("question_type") or "unspecified"),
        "question": str(sample.get("question") or ""),
        "document_ids": list(sample.get("document_ids") or []),
        "gold_evidence": gold_evidence,
        "gold_answer": str(sample.get("gold_answer") or ""),
        "gold_answer_keywords": list(sample.get("gold_answer_keywords") or []),
        "notes": str(sample.get("notes") or ""),
        "retrieval_mode": str(retrieval_timing.get("retrieval_mode") or "two_stage"),
        "guarded": int(bool(response_data.get("guarded"))),
        "guard_reason": str((response_data.get("guarded") or {}).get("reason") or ""),
        "raw_page_hit_at_1": hit_at_k(raw_predictions, gold_evidence, 1),
        "raw_page_hit_at_3": hit_at_k(raw_predictions, gold_evidence, 3),
        "raw_doc_hit_at_1": hit_at_k(raw_predictions, gold_evidence, 1, doc_only=True),
        "raw_doc_hit_at_3": hit_at_k(raw_predictions, gold_evidence, 3, doc_only=True),
        "raw_page_mrr": reciprocal_rank(raw_predictions, gold_evidence),
        "raw_doc_mrr": reciprocal_rank(raw_predictions, gold_evidence, doc_only=True),
        "raw_gold_rank": first_matching_rank(raw_predictions, gold_evidence),
        "final_page_hit_at_1": hit_at_k(final_predictions, gold_evidence, 1),
        "final_page_hit_at_3": hit_at_k(final_predictions, gold_evidence, 3),
        "final_doc_hit_at_1": hit_at_k(final_predictions, gold_evidence, 1, doc_only=True),
        "final_doc_hit_at_3": hit_at_k(final_predictions, gold_evidence, 3, doc_only=True),
        "final_page_mrr": reciprocal_rank(final_predictions, gold_evidence),
        "final_doc_mrr": reciprocal_rank(final_predictions, gold_evidence, doc_only=True),
        "final_gold_rank": first_matching_rank(final_predictions, gold_evidence),
        "answer_text": answer_text,
        "answer_has_citation": int(bool(citations)),
        "citation_count": len(citations),
        "citations": citations,
        "citation_has_gold_evidence": cited_gold_evidence,
        "answer_contains_gold_answer": compute_answer_string_match(answer_text, str(sample.get("gold_answer") or "")),
        **keyword_metrics,
        "automatic_faithfulness": automatic_faithfulness.get("score"),
        "automatic_faithfulness_label": automatic_faithfulness.get("label", ""),
        "automatic_faithfulness_reason": automatic_faithfulness.get("reason", ""),
        "automatic_faithfulness_provider": automatic_faithfulness.get("provider", ""),
        "confidence_label": str((response_data.get("confidence") or {}).get("label") or ""),
        "confidence_score": (response_data.get("confidence") or {}).get("score"),
        "selected_evidence_count": len(final_predictions),
        "raw_candidate_count": len(raw_predictions),
        "colpali_query_embedding_ms": float(retrieval_timing.get("colpali_query_embedding_ms") or 0.0),
        "muvera_query_embedding_ms": float(retrieval_timing.get("muvera_query_embedding_ms") or 0.0),
        "fallback_evidence_used": int(bool(retrieval_timing.get("fallback_evidence_used"))),
        "fallback_evidence_count": int(retrieval_timing.get("fallback_evidence_count") or 0),
        "fallback_best_score": float(retrieval_timing.get("fallback_best_score") or 0.0),
        "unsupported_sub_query_count": int(retrieval_timing.get("unsupported_sub_query_count") or 0),
        "reused_supported_sub_query_count": int(retrieval_timing.get("reused_supported_sub_query_count") or 0),
        "query_embedding_ms": float(retrieval_timing.get("query_embedding_ms") or 0.0),
        "qdrant_query_ms": float(retrieval_timing.get("qdrant_query_ms") or 0.0),
        "total_retrieval_ms": float(retrieval_timing.get("total_retrieval_ms") or 0.0),
        "client_first_token_ms": response_data.get("client_first_token_ms"),
        "client_total_chat_ms": response_data.get("client_total_chat_ms"),
        "returned_evidences": final_predictions,
        "all_candidates": raw_predictions,
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def numeric_average(field: str) -> Optional[float]:
        values = [float(record[field]) for record in records if record.get(field) not in (None, "")]
        return round(mean(values), 4) if values else None

    summary = {
        "sample_count": len(records),
        "guarded_count": int(sum(int(record.get("guarded") or 0) for record in records)),
        "retrieval_modes": sorted({str(record.get("retrieval_mode") or "") for record in records if str(record.get("retrieval_mode") or "")}),
        "raw_page_hit_at_1": numeric_average("raw_page_hit_at_1"),
        "raw_page_hit_at_3": numeric_average("raw_page_hit_at_3"),
        "raw_doc_hit_at_1": numeric_average("raw_doc_hit_at_1"),
        "raw_doc_hit_at_3": numeric_average("raw_doc_hit_at_3"),
        "raw_page_mrr": numeric_average("raw_page_mrr"),
        "final_page_hit_at_1": numeric_average("final_page_hit_at_1"),
        "final_page_hit_at_3": numeric_average("final_page_hit_at_3"),
        "final_doc_hit_at_1": numeric_average("final_doc_hit_at_1"),
        "final_doc_hit_at_3": numeric_average("final_doc_hit_at_3"),
        "final_page_mrr": numeric_average("final_page_mrr"),
        "fallback_rate": numeric_average("fallback_evidence_used"),
        "avg_unsupported_sub_query_count": numeric_average("unsupported_sub_query_count"),
        "avg_confidence_score": numeric_average("confidence_score"),
        "avg_colpali_query_embedding_ms": numeric_average("colpali_query_embedding_ms"),
        "avg_muvera_query_embedding_ms": numeric_average("muvera_query_embedding_ms"),
        "avg_total_retrieval_ms": numeric_average("total_retrieval_ms"),
        "avg_client_first_token_ms": numeric_average("client_first_token_ms"),
        "avg_client_total_chat_ms": numeric_average("client_total_chat_ms"),
        "answer_has_citation_rate": numeric_average("answer_has_citation"),
        "citation_has_gold_evidence_rate": numeric_average("citation_has_gold_evidence"),
        "answer_contains_gold_answer_rate": numeric_average("answer_contains_gold_answer"),
        "keyword_coverage_avg": numeric_average("keyword_coverage"),
        "keyword_full_match_rate": numeric_average("keyword_full_match"),
        "automatic_faithfulness_avg": numeric_average("automatic_faithfulness"),
    }
    summary["automatic_faithfulness_scored_samples"] = sum(
        1 for record in records if record.get("automatic_faithfulness") not in (None, "")
    )

    manual_fields = [
        "manual_answer_accuracy",
        "manual_faithfulness",
        "manual_citation_correctness",
    ]
    manual_scores = {
        field: [float(record[field]) for record in records if record.get(field) not in (None, "")]
        for field in manual_fields
    }
    summary["manual_reviewed_samples"] = max((len(values) for values in manual_scores.values()), default=0)
    for field, values in manual_scores.items():
        summary[f"{field}_avg"] = round(mean(values), 4) if values else None

    return summary


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark_id",
        "question_type",
        "question",
        "document_ids",
        "gold_evidence",
        "gold_answer",
        "gold_answer_keywords",
        "notes",
        "retrieval_mode",
        "guarded",
        "guard_reason",
        "raw_page_hit_at_1",
        "raw_page_hit_at_3",
        "raw_doc_hit_at_1",
        "raw_doc_hit_at_3",
        "raw_page_mrr",
        "raw_gold_rank",
        "final_page_hit_at_1",
        "final_page_hit_at_3",
        "final_doc_hit_at_1",
        "final_doc_hit_at_3",
        "final_page_mrr",
        "final_gold_rank",
        "fallback_evidence_used",
        "fallback_evidence_count",
        "fallback_best_score",
        "unsupported_sub_query_count",
        "reused_supported_sub_query_count",
        "confidence_label",
        "confidence_score",
        "colpali_query_embedding_ms",
        "muvera_query_embedding_ms",
        "query_embedding_ms",
        "qdrant_query_ms",
        "total_retrieval_ms",
        "client_first_token_ms",
        "client_total_chat_ms",
        "answer_has_citation",
        "citation_count",
        "citation_has_gold_evidence",
        "answer_contains_gold_answer",
        "keyword_coverage",
        "keyword_full_match",
        "automatic_faithfulness",
        "automatic_faithfulness_label",
        "automatic_faithfulness_reason",
        "automatic_faithfulness_provider",
        "matched_keywords",
        "citations",
        "top_selected_evidence",
        "top_raw_candidates",
        "answer_text",
        "manual_answer_accuracy",
        "manual_faithfulness",
        "manual_citation_correctness",
        "reviewer_notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "benchmark_id": record.get("benchmark_id", ""),
                    "question_type": record.get("question_type", ""),
                    "question": record.get("question", ""),
                    "document_ids": serialize_json(record.get("document_ids") or []),
                    "gold_evidence": serialize_json(record.get("gold_evidence") or []),
                    "gold_answer": record.get("gold_answer", ""),
                    "gold_answer_keywords": serialize_json(record.get("gold_answer_keywords") or []),
                    "notes": record.get("notes", ""),
                    "retrieval_mode": record.get("retrieval_mode", ""),
                    "guarded": record.get("guarded", ""),
                    "guard_reason": record.get("guard_reason", ""),
                    "raw_page_hit_at_1": record.get("raw_page_hit_at_1", ""),
                    "raw_page_hit_at_3": record.get("raw_page_hit_at_3", ""),
                    "raw_doc_hit_at_1": record.get("raw_doc_hit_at_1", ""),
                    "raw_doc_hit_at_3": record.get("raw_doc_hit_at_3", ""),
                    "raw_page_mrr": record.get("raw_page_mrr", ""),
                    "raw_gold_rank": record.get("raw_gold_rank", ""),
                    "final_page_hit_at_1": record.get("final_page_hit_at_1", ""),
                    "final_page_hit_at_3": record.get("final_page_hit_at_3", ""),
                    "final_doc_hit_at_1": record.get("final_doc_hit_at_1", ""),
                    "final_doc_hit_at_3": record.get("final_doc_hit_at_3", ""),
                    "final_page_mrr": record.get("final_page_mrr", ""),
                    "final_gold_rank": record.get("final_gold_rank", ""),
                    "fallback_evidence_used": record.get("fallback_evidence_used", ""),
                    "fallback_evidence_count": record.get("fallback_evidence_count", ""),
                    "fallback_best_score": record.get("fallback_best_score", ""),
                    "unsupported_sub_query_count": record.get("unsupported_sub_query_count", ""),
                    "reused_supported_sub_query_count": record.get("reused_supported_sub_query_count", ""),
                    "confidence_label": record.get("confidence_label", ""),
                    "confidence_score": record.get("confidence_score", ""),
                    "colpali_query_embedding_ms": record.get("colpali_query_embedding_ms", ""),
                    "muvera_query_embedding_ms": record.get("muvera_query_embedding_ms", ""),
                    "query_embedding_ms": record.get("query_embedding_ms", ""),
                    "qdrant_query_ms": record.get("qdrant_query_ms", ""),
                    "total_retrieval_ms": record.get("total_retrieval_ms", ""),
                    "client_first_token_ms": record.get("client_first_token_ms", ""),
                    "client_total_chat_ms": record.get("client_total_chat_ms", ""),
                    "answer_has_citation": record.get("answer_has_citation", ""),
                    "citation_count": record.get("citation_count", ""),
                    "citation_has_gold_evidence": record.get("citation_has_gold_evidence", ""),
                    "answer_contains_gold_answer": record.get("answer_contains_gold_answer", ""),
                    "keyword_coverage": record.get("keyword_coverage", ""),
                    "keyword_full_match": record.get("keyword_full_match", ""),
                    "automatic_faithfulness": record.get("automatic_faithfulness", ""),
                    "automatic_faithfulness_label": record.get("automatic_faithfulness_label", ""),
                    "automatic_faithfulness_reason": record.get("automatic_faithfulness_reason", ""),
                    "automatic_faithfulness_provider": record.get("automatic_faithfulness_provider", ""),
                    "matched_keywords": serialize_json(record.get("matched_keywords") or []),
                    "citations": serialize_json(record.get("citations") or []),
                    "top_selected_evidence": format_evidence_list(record.get("returned_evidences") or []),
                    "top_raw_candidates": format_evidence_list(record.get("all_candidates") or []),
                    "answer_text": record.get("answer_text", ""),
                    "manual_answer_accuracy": "",
                    "manual_faithfulness": "",
                    "manual_citation_correctness": "",
                    "reviewer_notes": "",
                }
            )


def parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def summarize_review_csv(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {
                "benchmark_id": row.get("benchmark_id", ""),
                "retrieval_mode": row.get("retrieval_mode", "two_stage"),
                "raw_page_hit_at_1": parse_optional_float(row.get("raw_page_hit_at_1")),
                "raw_page_hit_at_3": parse_optional_float(row.get("raw_page_hit_at_3")),
                "raw_doc_hit_at_1": parse_optional_float(row.get("raw_doc_hit_at_1")),
                "raw_doc_hit_at_3": parse_optional_float(row.get("raw_doc_hit_at_3")),
                "raw_page_mrr": parse_optional_float(row.get("raw_page_mrr")),
                "final_page_hit_at_1": parse_optional_float(row.get("final_page_hit_at_1")),
                "final_page_hit_at_3": parse_optional_float(row.get("final_page_hit_at_3")),
                "final_doc_hit_at_1": parse_optional_float(row.get("final_doc_hit_at_1")),
                "final_doc_hit_at_3": parse_optional_float(row.get("final_doc_hit_at_3")),
                "final_page_mrr": parse_optional_float(row.get("final_page_mrr")),
                "fallback_evidence_used": parse_optional_float(row.get("fallback_evidence_used")),
                "unsupported_sub_query_count": parse_optional_float(row.get("unsupported_sub_query_count")),
                "confidence_score": parse_optional_float(row.get("confidence_score")),
                "colpali_query_embedding_ms": parse_optional_float(row.get("colpali_query_embedding_ms")),
                "muvera_query_embedding_ms": parse_optional_float(row.get("muvera_query_embedding_ms")),
                "total_retrieval_ms": parse_optional_float(row.get("total_retrieval_ms")),
                "client_first_token_ms": parse_optional_float(row.get("client_first_token_ms")),
                "client_total_chat_ms": parse_optional_float(row.get("client_total_chat_ms")),
                "answer_has_citation": parse_optional_float(row.get("answer_has_citation")),
                "citation_has_gold_evidence": parse_optional_float(row.get("citation_has_gold_evidence")),
                "answer_contains_gold_answer": parse_optional_float(row.get("answer_contains_gold_answer")),
                "keyword_coverage": parse_optional_float(row.get("keyword_coverage")),
                "keyword_full_match": parse_optional_float(row.get("keyword_full_match")),
                "automatic_faithfulness": parse_optional_float(row.get("automatic_faithfulness")),
                "manual_answer_accuracy": parse_optional_float(row.get("manual_answer_accuracy")),
                "manual_faithfulness": parse_optional_float(row.get("manual_faithfulness")),
                "manual_citation_correctness": parse_optional_float(row.get("manual_citation_correctness")),
                "guarded": parse_optional_float(row.get("guarded")) or 0.0,
            }
            rows.append(parsed)
    return summarize_records(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    print("RAG evaluation summary")
    if summary.get("retrieval_modes"):
        print(f"- retrieval modes: {', '.join(summary.get('retrieval_modes') or [])}")
    print(f"- samples: {summary.get('sample_count')}")
    print(f"- raw page Hit@1: {format_rate(summary.get('raw_page_hit_at_1'))}")
    print(f"- raw page Hit@3: {format_rate(summary.get('raw_page_hit_at_3'))}")
    print(f"- raw page MRR: {format_decimal(summary.get('raw_page_mrr'))}")
    print(f"- final page Hit@1: {format_rate(summary.get('final_page_hit_at_1'))}")
    print(f"- final page Hit@3: {format_rate(summary.get('final_page_hit_at_3'))}")
    print(f"- final page MRR: {format_decimal(summary.get('final_page_mrr'))}")
    print(f"- avg ColPali query embed ms: {format_decimal(summary.get('avg_colpali_query_embedding_ms'))}")
    print(f"- avg MUVERA query embed ms: {format_decimal(summary.get('avg_muvera_query_embedding_ms'))}")
    print(f"- avg retrieval ms: {format_decimal(summary.get('avg_total_retrieval_ms'))}")
    print(f"- avg first token ms: {format_decimal(summary.get('avg_client_first_token_ms'))}")
    print(f"- avg total chat ms: {format_decimal(summary.get('avg_client_total_chat_ms'))}")
    print(f"- fallback rate: {format_rate(summary.get('fallback_rate'))}")
    print(f"- answer has citation rate: {format_rate(summary.get('answer_has_citation_rate'))}")
    print(f"- citation hits gold evidence rate: {format_rate(summary.get('citation_has_gold_evidence_rate'))}")
    if summary.get("automatic_faithfulness_scored_samples"):
        print(f"- automatic faithfulness avg: {format_decimal(summary.get('automatic_faithfulness_avg'))}")
    if summary.get("manual_reviewed_samples"):
        print(f"- manual answer accuracy avg: {format_decimal(summary.get('manual_answer_accuracy_avg'))}")
        print(f"- manual faithfulness avg: {format_decimal(summary.get('manual_faithfulness_avg'))}")
        print(f"- manual citation correctness avg: {format_decimal(summary.get('manual_citation_correctness_avg'))}")


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_decimal(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def main() -> int:
    args = parse_args()

    if args.review_csv:
        review_path = Path(args.review_csv)
        summary = summarize_review_csv(review_path)
        print_summary(summary)
        write_json(review_path.with_name("review_summary.json"), summary)
        return 0

    if not args.benchmark_file:
        raise SystemExit("--benchmark-file is required unless --review-csv is used.")

    benchmark_path = Path(args.benchmark_file)
    benchmark_payload = load_benchmark(benchmark_path)
    defaults = dict(benchmark_payload.get("defaults") or {})
    benchmark_items = list(benchmark_payload.get("items") or [])
    if args.limit:
        benchmark_items = benchmark_items[: args.limit]

    output_dir = build_output_dir(args.output_dir)
    write_json(output_dir / "benchmark_snapshot.json", benchmark_payload)

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(benchmark_items, start=1):
        question_id = str(sample.get("id") or f"item-{index}")
        print(f"[{index}/{len(benchmark_items)}] Running {question_id} ...")
        response_data = run_chat_sample(
            api_base_url=args.api_base_url,
            question=str(sample.get("question") or ""),
            document_ids=list(sample.get("document_ids") or []),
            top_k=int(args.top_k or sample.get("top_k") or defaults.get("top_k") or 5),
            min_score=float(args.min_score or sample.get("min_score") or defaults.get("min_score") or 0.6),
            retrieval_mode=str(args.retrieval_mode or "two_stage"),
            chat_history=list(sample.get("chat_history") or defaults.get("chat_history") or []),
            timeout_seconds=float(args.timeout_seconds),
        )
        record = evaluate_sample(
            sample,
            response_data,
            enable_automatic_faithfulness=not args.disable_automatic_faithfulness,
            faithfulness_model_name=args.faithfulness_model_name,
        )
        records.append(record)

    summary = summarize_records(records)
    write_json(output_dir / "detailed_results.json", records)
    write_json(output_dir / "summary.json", summary)
    write_review_csv(output_dir / "review_sheet.csv", records)
    print_summary(summary)
    print(f"Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())