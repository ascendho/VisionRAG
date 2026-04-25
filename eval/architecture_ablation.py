"""
Architecture ablation benchmark for VisionRAG.

无需 gold-page 标注；所有消融指标基于系统延迟、吞吐、检索分布与缓存行为。

支持两个 track：
  upload  — 上传链路消融（dpi / embed_batch_size / upsert_batch_size / muvera_r_reps / muvera_dim_proj）
  answer  — 问答链路消融（prefetch_multiplier / top_k / min_score / max_tokens）

用法示例：
  # 问答链路：扫描 prefetch_multiplier
  python -m eval.architecture_ablation --track answer \\
      --knob prefetch_multiplier --values "1,5,10,20,50" \\
      --queries eval/queries/answer_queries.jsonl --runs 3

  # 上传链路：扫描 dpi
  python -m eval.architecture_ablation --track upload \\
      --knob dpi --values "75,150,300" \\
      --input qdrant_local/pdfs/mypdf.pdf --runs 3

每次只改一个 knob，与 baseline（第一个值或 --baseline 指定的值）对比。
结果写入 eval/results/ablation_<track>_<knob>_<timestamp>.jsonl
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import IMAGE_CACHE_DIR
from src.doc_processor import get_file_hash, process_pdf_to_images
from src.vector_store import VisionVectorStore
from eval.upload_benchmark import (
    _benchmark_single_run as _upload_single_run,
    _build_summary as _upload_build_summary,
    _collect_pdf_files,
    _write_jsonl,
)
from eval.answer_benchmark import (
    _load_queries,
    _build_summary as _answer_build_summary,
    run_answer_benchmark,
)

# ── Knob registry ─────────────────────────────────────────────────────────────

UPLOAD_KNOBS = {
    "dpi",
    "embed_batch_size",
    "upsert_batch_size",
    "muvera_r_reps",
    "muvera_dim_proj",
}

ANSWER_KNOBS = {
    "prefetch_multiplier",
    "top_k",
    "min_score",
    "max_tokens",
}

MUVERA_REINIT_KNOBS = {"muvera_r_reps", "muvera_dim_proj"}

# Default baseline values for each knob
KNOB_DEFAULTS: Dict[str, Any] = {
    "dpi": 150,
    "embed_batch_size": 4,
    "upsert_batch_size": 8,
    "muvera_r_reps": 30,
    "muvera_dim_proj": 16,
    "prefetch_multiplier": 10,
    "top_k": 5,
    "min_score": 0.6,
    "max_tokens": 800,
}


def _cast_value(knob: str, raw: str) -> Any:
    """Cast a string knob value to the appropriate Python type."""
    if knob == "min_score":
        return float(raw)
    return int(raw)


def _parse_values(knob: str, raw: str) -> List[Any]:
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    return [_cast_value(knob, p) for p in parts]


# ── Per-run helpers ───────────────────────────────────────────────────────────

def _run_upload_variant(
    knob: str,
    value: Any,
    pdfs: List[Path],
    runs: int,
    store: VisionVectorStore,
) -> List[Dict[str, Any]]:
    """Run upload benchmark for one knob variant, returning raw result rows."""
    dpi = value if knob == "dpi" else KNOB_DEFAULTS["dpi"]
    embed_batch_size = value if knob == "embed_batch_size" else KNOB_DEFAULTS["embed_batch_size"]
    upsert_batch_size = value if knob == "upsert_batch_size" else KNOB_DEFAULTS["upsert_batch_size"]

    results: List[Dict[str, Any]] = []
    for pdf_path in pdfs:
        for run_index in range(1, runs + 1):
            for cache_mode in ("cold", "hot"):
                if cache_mode == "cold":
                    cache_dir = Path(IMAGE_CACHE_DIR) / get_file_hash(str(pdf_path))
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                else:
                    process_pdf_to_images(str(pdf_path), dpi=dpi)

                total_t0 = time.perf_counter()
                image_paths, prepare_timing = process_pdf_to_images(
                    str(pdf_path), dpi=dpi, return_timing=True
                )
                success, index_timing = store.embed_and_store_documents(
                    image_paths=image_paths,
                    document_id=get_file_hash(str(pdf_path)),
                    document_name=pdf_path.name,
                    batch_size=embed_batch_size,
                    upsert_batch_size=upsert_batch_size,
                    return_timing=True,
                )
                total_index_ms = round((time.perf_counter() - total_t0) * 1000, 1)
                if not success:
                    print(f"  [warning] indexing returned False for {pdf_path.name}")
                    continue

                page_count = int(index_timing.get("page_count", len(image_paths)) or len(image_paths))
                pages_per_second = round(page_count / max(total_index_ms / 1000, 1e-6), 2)
                row: Dict[str, Any] = {
                    "variant": f"{knob}={value}",
                    "knob": knob,
                    "value": value,
                    "file_name": pdf_path.name,
                    "run_index": run_index,
                    "cache_mode": cache_mode,
                    "cache_hit": bool(prepare_timing.get("cache_hit", False)),
                    "page_count": page_count,
                    "pdf_render_ms": float(prepare_timing.get("pdf_render_ms", 0.0)),
                    "cache_write_ms": float(prepare_timing.get("cache_write_ms", 0.0)),
                    "total_prepare_ms": float(prepare_timing.get("total_prepare_ms", 0.0)),
                    "embedding_ms": float(index_timing.get("embedding_ms", 0.0)),
                    "compression_ms": float(index_timing.get("compression_ms", 0.0)),
                    "upsert_ms": float(index_timing.get("upsert_ms", 0.0)),
                    "total_index_ms": total_index_ms,
                    "pages_per_second": pages_per_second,
                }
                print(
                    f"  [{cache_mode}] {knob}={value} run={run_index} "
                    f"pages={page_count} total={total_index_ms:.1f}ms "
                    f"render={row['pdf_render_ms']:.1f}ms embed={row['embedding_ms']:.1f}ms "
                    f"upsert={row['upsert_ms']:.1f}ms"
                )
                results.append(row)
    return results


def _run_answer_variant(
    knob: str,
    value: Any,
    queries: List[Dict[str, Any]],
    runs: int,
    store: VisionVectorStore,
) -> List[Dict[str, Any]]:
    """Run answer benchmark for one knob variant, returning raw result rows."""
    prefetch_multiplier = value if knob == "prefetch_multiplier" else KNOB_DEFAULTS["prefetch_multiplier"]
    top_k = value if knob == "top_k" else KNOB_DEFAULTS["top_k"]
    min_score = value if knob == "min_score" else KNOB_DEFAULTS["min_score"]
    max_tokens = value if knob == "max_tokens" else KNOB_DEFAULTS["max_tokens"]

    raw = run_answer_benchmark(
        store=store,
        queries=queries,
        runs=runs,
        top_k=top_k,
        prefetch_multiplier=prefetch_multiplier,
        min_score=min_score,
        max_tokens=max_tokens,
        output_dir=Path("eval/results"),
        label=f"{knob}={value}",
    )
    for row in raw:
        row["variant"] = f"{knob}={value}"
        row["knob"] = knob
        row["value"] = value
    return raw


# ── Summary helpers ───────────────────────────────────────────────────────────

_UPLOAD_REPORT_FIELDS = (
    "pdf_render_ms",
    "embedding_ms",
    "upsert_ms",
    "total_index_ms",
    "pages_per_second",
)

_ANSWER_REPORT_FIELDS = (
    "time_to_evidence_ms",
    "time_to_first_token_ms",
    "total_answer_latency_ms",
    "tokens_per_second",
)


def _print_comparison(
    track: str,
    knob: str,
    values: List[Any],
    all_variant_results: Dict[Any, List[Dict[str, Any]]],
) -> None:
    report_fields = _UPLOAD_REPORT_FIELDS if track == "upload" else _ANSWER_REPORT_FIELDS
    build_summary = _upload_build_summary if track == "upload" else _answer_build_summary

    print(f"\n架构消融汇总 — track={track}  knob={knob}")
    print("=" * 80)

    for mode in ("cold", "hot"):
        mode_rows: List[Dict[str, Any]] = []
        for v in values:
            rows = [r for r in all_variant_results[v] if r.get("cache_mode") == mode]
            mode_rows.append((v, rows))

        if not any(rows for _, rows in mode_rows):
            continue

        print(f"\n[{mode}]")
        header = f"  {'variant':<25}" + "".join(f"  {f[:16]:<16}" for f in report_fields)
        print(header)
        print("  " + "-" * (24 + 18 * len(report_fields)))

        for v, rows in mode_rows:
            by_mode = build_summary(rows)
            stats_for_mode = by_mode.get(mode, {})
            cells = []
            for f in report_fields:
                p50 = stats_for_mode.get(f, {}).get("p50", float("nan"))
                cells.append(f"  {p50:>14.1f}  ")
            print(f"  {f'{knob}={v}':<25}" + "".join(cells))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VisionRAG 架构消融 benchmark（无 gold-page）")
    parser.add_argument(
        "--track",
        required=True,
        choices=["upload", "answer"],
        help="消融 track：upload（上传链路）或 answer（问答链路）",
    )
    parser.add_argument(
        "--knob",
        required=True,
        help=(
            f"要消融的参数名。upload 支持：{', '.join(sorted(UPLOAD_KNOBS))}；"
            f"answer 支持：{', '.join(sorted(ANSWER_KNOBS))}"
        ),
    )
    parser.add_argument(
        "--values",
        required=True,
        help="逗号分隔的候选值列表，第一个值视为 baseline，例如 '1,5,10,20'",
    )
    parser.add_argument("--runs", type=int, default=3, help="每个 variant 每条查询/文件的重复次数（默认 3）")
    # upload-specific
    parser.add_argument("--input", help="（upload track）PDF 文件或目录")
    # answer-specific
    parser.add_argument("--queries", help="（answer track）查询文件路径（JSONL）")
    # muvera reinit
    parser.add_argument(
        "--muvera-r-reps", type=int, default=30,
        help="MUVERA r_reps 基准值（muvera_r_reps 消融时忽略此值，自动扫描 --values）"
    )
    parser.add_argument(
        "--muvera-dim-proj", type=int, default=16,
        help="MUVERA dim_proj 基准值（muvera_dim_proj 消融时忽略此值，自动扫描 --values）"
    )
    parser.add_argument("--output", default="eval/results", help="结果输出目录（默认 eval/results）")
    args = parser.parse_args()

    # Validate knob vs track
    if args.track == "upload" and args.knob not in UPLOAD_KNOBS:
        raise SystemExit(f"错误：knob '{args.knob}' 不属于 upload track，可选：{sorted(UPLOAD_KNOBS)}")
    if args.track == "answer" and args.knob not in ANSWER_KNOBS:
        raise SystemExit(f"错误：knob '{args.knob}' 不属于 answer track，可选：{sorted(ANSWER_KNOBS)}")
    if args.track == "upload" and not args.input:
        raise SystemExit("错误：upload track 需要 --input")
    if args.track == "answer" and not args.queries:
        raise SystemExit("错误：answer track 需要 --queries")
    if args.runs <= 0:
        raise SystemExit("错误：--runs 必须为正整数")

    values = _parse_values(args.knob, args.values)
    if not values:
        raise SystemExit("错误：--values 解析结果为空")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare inputs
    pdfs: Optional[List[Path]] = None
    queries: Optional[List[Dict[str, Any]]] = None
    if args.track == "upload":
        pdfs = _collect_pdf_files(args.input)
    else:
        queries = _load_queries(args.queries)

    print(f"正在加载向量库与模型（首次约需 1–2 分钟）…")

    all_variant_results: Dict[Any, List[Dict[str, Any]]] = {}
    all_rows: List[Dict[str, Any]] = []

    for value in values:
        print(f"\n── variant: {args.knob}={value} ──────────────────────────────────────────")

        # MUVERA param knobs require re-instantiation
        if args.knob in MUVERA_REINIT_KNOBS:
            r_reps = value if args.knob == "muvera_r_reps" else args.muvera_r_reps
            dim_proj = value if args.knob == "muvera_dim_proj" else args.muvera_dim_proj
            print(f"  初始化 VisionVectorStore（r_reps={r_reps}, dim_proj={dim_proj}）…")
            store = VisionVectorStore(muvera_r_reps=r_reps, muvera_dim_proj=dim_proj)
        else:
            # Reuse single store for all non-MUVERA variants
            if value == values[0] or args.knob in MUVERA_REINIT_KNOBS:
                print("  初始化 VisionVectorStore…")
                store = VisionVectorStore()

        if args.track == "upload":
            rows = _run_upload_variant(
                knob=args.knob,
                value=value,
                pdfs=pdfs,
                runs=args.runs,
                store=store,
            )
        else:
            rows = _run_answer_variant(
                knob=args.knob,
                value=value,
                queries=queries,
                runs=args.runs,
                store=store,
            )

        all_variant_results[value] = rows
        all_rows.extend(rows)

    # Print comparison table
    _print_comparison(args.track, args.knob, values, all_variant_results)

    # Build and write output
    ablation_summary = {
        "_type": "ablation_summary",
        "timestamp": timestamp,
        "track": args.track,
        "knob": args.knob,
        "values": values,
        "baseline_value": values[0],
        "config": {
            "runs": args.runs,
            "input": str(pdfs[0]) if pdfs else None,
            "queries": args.queries,
        },
        "by_variant": {
            str(v): {
                "by_mode": (
                    _upload_build_summary(all_variant_results[v])
                    if args.track == "upload"
                    else _answer_build_summary(all_variant_results[v])
                )
            }
            for v in values
        },
    }

    output_path = output_dir / f"ablation_{args.track}_{args.knob}_{timestamp}.jsonl"
    _write_jsonl(output_path, [ablation_summary] + all_rows)
    print(f"\n消融结果已保存：{output_path}")


if __name__ == "__main__":
    main()
