"""
Answer/query benchmark for VisionRAG.

第二阶段只测问答链路，不依赖 gold pages。

测量项：
- 查询 embedding（query_embedding_ms）
- MUVERA 查询侧压缩（muvera_query_ms）
- Qdrant 检索（qdrant_query_ms）
- 后处理（postprocess_ms）
- 检索总耗时 / 首帧证据时延（time_to_evidence_ms）
- 首 token 时延（time_to_first_token_ms）
- 生成总耗时（total_generation_ms）
- 完整问答链路耗时（total_answer_latency_ms）
- 输出 token 数量与速率

cold / hot 定义：
  cold = run_index 1，即查询序列的第一轮（Qdrant 页缓存未预热）
  hot  = run_index 2..N（模型 warm、Qdrant 内存页已缓存）

结果按 cold / hot 分组汇总 p50 / p95，并写入 JSONL。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.doc_processor import get_file_hash
from src.llm_generator import generate_answer_stream
from src.vector_store import VisionVectorStore

SUMMARY_FIELDS = [
    "query_embedding_ms",
    "muvera_query_ms",
    "qdrant_query_ms",
    "postprocess_ms",
    "retrieval_total_ms",
    "time_to_evidence_ms",
    "time_to_first_token_ms",
    "total_generation_ms",
    "total_answer_latency_ms",
    "retrieved_count",
    "accepted_count",
    "token_count",
    "tokens_per_second",
]


def _load_queries(query_path: str) -> List[Dict[str, Any]]:
    path = Path(query_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"查询文件不存在：{query_path}")
    queries = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"查询文件第 {lineno} 行 JSON 解析失败：{exc}") from exc
            if "query" not in row:
                raise ValueError(f"查询文件第 {lineno} 行缺少 'query' 字段")
            if "query_id" not in row:
                row["query_id"] = f"q{lineno:03d}"
            queries.append(row)
    if not queries:
        raise ValueError("查询文件为空，至少需要一条查询。")
    return queries


def _resolve_document_ids(
    store: VisionVectorStore,
    document_name: Optional[str],
) -> Optional[List[str]]:
    if not document_name:
        return None
    docs = store.get_all_documents()
    matched = [d["document_id"] for d in docs if d.get("document_name") == document_name]
    if not matched:
        available = [d.get("document_name") for d in docs]
        raise ValueError(
            f"找不到文档 '{document_name}'，当前已索引：{available}"
        )
    return matched


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 1)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "min": round(ordered[0], 1),
        "max": round(ordered[-1], 1),
    }


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in sorted({r["cache_mode"] for r in results}):
        mode_results = [r for r in results if r["cache_mode"] == mode]
        summary[mode] = {
            field: _summarize([float(r.get(field, 0)) for r in mode_results])
            for field in SUMMARY_FIELDS
        }
    return summary


def _benchmark_single_query(
    store: VisionVectorStore,
    query: Dict[str, Any],
    run_index: int,
    top_k: int,
    prefetch_multiplier: int,
    min_score: float,
    max_tokens: int,
) -> Dict[str, Any]:
    cache_mode = "cold" if run_index == 1 else "hot"

    # Resolve optional document filter
    document_ids = _resolve_document_ids(store, query.get("document_name"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results, retrieval_timing = store.retrieve_with_two_stage(
        query_text=query["query"],
        top_k=top_k,
        prefetch_multiplier=prefetch_multiplier,
        document_ids=document_ids,
        return_timing=True,
    )
    t_evidence = time.perf_counter()

    # Apply min_score filter (mirrors what the API route does)
    accepted = [r for r in results if float(r.get("score", 0)) >= min_score]
    if not accepted and results:
        accepted = results[:1]
    evidence_paths = [r["image_path"] for r in accepted if r.get("image_path")]

    scores = [float(r.get("score", 0)) for r in results]
    score_p50 = _percentile(scores, 0.50)
    score_p95 = _percentile(scores, 0.95)

    time_to_evidence_ms = round((t_evidence - t_start) * 1000, 1)

    # ── Generation ───────────────────────────────────────────────────────────
    t_gen_start = time.perf_counter()
    first_token_ms: Optional[float] = None
    token_count = 0

    if evidence_paths:
        for token in generate_answer_stream(
            query_text=query["query"],
            image_paths=evidence_paths,
            max_tokens=max_tokens,
        ):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t_gen_start) * 1000, 1)
            token_count += len(token)
    else:
        first_token_ms = 0.0

    t_end = time.perf_counter()
    total_generation_ms = round((t_end - t_gen_start) * 1000, 1)
    total_answer_latency_ms = round((t_end - t_start) * 1000, 1)
    tokens_per_second = round(
        token_count / max(total_generation_ms / 1000, 1e-6), 2
    ) if token_count else 0.0

    return {
        "query_id": query.get("query_id", ""),
        "query": query["query"],
        "document_name": query.get("document_name"),
        "run_index": run_index,
        "cache_mode": cache_mode,
        "top_k": top_k,
        "prefetch_multiplier": prefetch_multiplier,
        "min_score": min_score,
        "max_tokens": max_tokens,
        "retrieved_count": len(results),
        "accepted_count": len(accepted),
        "score_p50": score_p50,
        "score_p95": score_p95,
        "query_embedding_ms": float(retrieval_timing.get("query_embedding_ms", 0)),
        "muvera_query_ms": float(retrieval_timing.get("muvera_query_ms", 0)),
        "qdrant_query_ms": float(retrieval_timing.get("qdrant_query_ms", 0)),
        "postprocess_ms": float(retrieval_timing.get("postprocess_ms", 0)),
        "retrieval_total_ms": float(retrieval_timing.get("total_ms", 0)),
        "time_to_evidence_ms": time_to_evidence_ms,
        "time_to_first_token_ms": float(first_token_ms if first_token_ms is not None else 0.0),
        "total_generation_ms": total_generation_ms,
        "total_answer_latency_ms": total_answer_latency_ms,
        "token_count": token_count,
        "tokens_per_second": tokens_per_second,
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_answer_benchmark(
    store: VisionVectorStore,
    queries: List[Dict[str, Any]],
    *,
    runs: int = 3,
    top_k: int = 5,
    prefetch_multiplier: int = 10,
    min_score: float = 0.6,
    max_tokens: int = 800,
    output_dir: Path,
    label: str = "",
) -> List[Dict[str, Any]]:
    """Run the full query benchmark and return raw result rows.

    Exposed as a function so the architecture ablation runner can call it
    directly with different parameter variants.
    """
    results: List[Dict[str, Any]] = []
    for run_index in range(1, runs + 1):
        for query in queries:
            result = _benchmark_single_query(
                store=store,
                query=query,
                run_index=run_index,
                top_k=top_k,
                prefetch_multiplier=prefetch_multiplier,
                min_score=min_score,
                max_tokens=max_tokens,
            )
            variant = f"[{result['cache_mode']}]{' ' + label if label else ''}"
            print(
                f"  {variant} {query.get('query_id', '')} run={run_index} "
                f"evidence={result['time_to_evidence_ms']:.1f}ms "
                f"ttft={result['time_to_first_token_ms']:.1f}ms "
                f"total={result['total_answer_latency_ms']:.1f}ms "
                f"tokens={result['token_count']}"
            )
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionRAG 问答链路 benchmark")
    parser.add_argument("--queries", required=True, help="查询文件路径（JSONL）")
    parser.add_argument("--runs", type=int, default=3, help="每条查询的重复次数（默认 3；run 1=cold，run 2+=hot）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回最多页数（默认 5）")
    parser.add_argument("--prefetch-multiplier", type=int, default=10, help="MUVERA 海选倍率（默认 10）")
    parser.add_argument("--min-score", type=float, default=0.6, help="证据接受最低分（默认 0.6）")
    parser.add_argument("--max-tokens", type=int, default=800, help="生成最大 token 数（默认 800）")
    parser.add_argument("--output", default="eval/results", help="结果输出目录（默认 eval/results）")
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("错误：--runs 必须为正整数")

    queries = _load_queries(args.queries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载向量库与模型（首次约需 1–2 分钟）…")
    store = VisionVectorStore()

    print(
        f"开始问答链路 benchmark：{len(queries)} 条查询，"
        f"runs={args.runs}，top_k={args.top_k}，"
        f"prefetch_multiplier={args.prefetch_multiplier}，"
        f"min_score={args.min_score}，max_tokens={args.max_tokens}"
    )
    results = run_answer_benchmark(
        store=store,
        queries=queries,
        runs=args.runs,
        top_k=args.top_k,
        prefetch_multiplier=args.prefetch_multiplier,
        min_score=args.min_score,
        max_tokens=args.max_tokens,
        output_dir=output_dir,
    )

    summary = {
        "_type": "summary",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "answer",
        "config": {
            "queries": str(Path(args.queries).expanduser()),
            "runs": args.runs,
            "top_k": args.top_k,
            "prefetch_multiplier": args.prefetch_multiplier,
            "min_score": args.min_score,
            "max_tokens": args.max_tokens,
            "query_count": len(queries),
        },
        "by_mode": _build_summary(results),
    }

    timestamp = summary["timestamp"]
    output_path = output_dir / f"answer_benchmark_{timestamp}.jsonl"
    _write_jsonl(output_path, [summary] + results)

    print("\n问答链路汇总（按 cache_mode）")
    print("=" * 72)
    for mode, mode_summary in summary["by_mode"].items():
        print(f"[{mode}]")
        for field in (
            "time_to_evidence_ms",
            "time_to_first_token_ms",
            "total_answer_latency_ms",
            "tokens_per_second",
        ):
            stats = mode_summary.get(field, {})
            if stats:
                print(
                    f"  {field:<24} p50={stats['p50']:>8.1f}  "
                    f"p95={stats['p95']:>8.1f}  mean={stats['mean']:>8.1f}"
                )
        print()

    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()
