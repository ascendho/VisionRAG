"""
Upload/index benchmark for VisionRAG.

第一阶段只测上传链路，不测问答链路，也不依赖 gold pages。

测量项：
- PDF 渲染（pdf_render_ms）
- 嵌入计算（embedding_ms）
- MUVERA 压缩（compression_ms）
- Qdrant upsert（upsert_ms）
- 总索引时间（total_index_ms）
- 吞吐（pages_per_second）

输出按 cold / hot 分组汇总 p50 / p95，并写入 JSONL。
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import IMAGE_CACHE_DIR
from src.doc_processor import get_file_hash, process_pdf_to_images
from src.vector_store import VisionVectorStore


SUMMARY_FIELDS = [
    "pdf_render_ms",
    "cache_write_ms",
    "total_prepare_ms",
    "embedding_ms",
    "compression_ms",
    "upsert_ms",
    "total_index_ms",
    "pages_per_second",
]


def _collect_pdf_files(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError("第一版上传 benchmark 仅支持 PDF 文件。")
        return [path.resolve()]
    pdfs = sorted(file.resolve() for file in path.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"目录中未找到 PDF 文件：{input_path}")
    return pdfs


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    position = min(max(q, 0.0), 1.0) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    interpolated = ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
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
    for mode in sorted({result["cache_mode"] for result in results}):
        mode_results = [result for result in results if result["cache_mode"] == mode]
        summary[mode] = {
            field: _summarize([float(result[field]) for result in mode_results])
            for field in SUMMARY_FIELDS
        }
    return summary


def _remove_document_cache(pdf_path: Path) -> None:
    cache_dir = Path(IMAGE_CACHE_DIR) / get_file_hash(str(pdf_path))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _ensure_hot_cache(pdf_path: Path) -> None:
    process_pdf_to_images(str(pdf_path))


def _benchmark_single_run(
    store: VisionVectorStore,
    pdf_path: Path,
    cache_mode: str,
    run_index: int,
) -> Dict[str, Any]:
    if cache_mode == "cold":
        _remove_document_cache(pdf_path)
    elif cache_mode == "hot":
        _ensure_hot_cache(pdf_path)
    else:
        raise ValueError(f"未知缓存模式：{cache_mode}")

    total_t0 = time.perf_counter()
    image_paths, prepare_timing = process_pdf_to_images(str(pdf_path), return_timing=True)
    success, index_timing = store.embed_and_store_documents(
        image_paths=image_paths,
        document_id=get_file_hash(str(pdf_path)),
        document_name=pdf_path.name,
        return_timing=True,
    )
    total_index_ms = round((time.perf_counter() - total_t0) * 1000, 1)
    if not success:
        raise RuntimeError(f"文档索引失败：{pdf_path}")

    page_count = int(index_timing.get("page_count", len(image_paths)) or len(image_paths))
    pages_per_second = round(page_count / max(total_index_ms / 1000, 1e-6), 2)
    return {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 3),
        "document_id": get_file_hash(str(pdf_path)),
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


def _iter_modes(mode: str) -> Iterable[str]:
    if mode == "both":
        return ("cold", "hot")
    return (mode,)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionRAG 上传链路 benchmark（PDF）")
    parser.add_argument("--input", required=True, help="单个 PDF 文件或包含 PDF 的目录")
    parser.add_argument(
        "--mode",
        choices=["cold", "hot", "both"],
        default="both",
        help="缓存模式：cold / hot / both（默认 both）",
    )
    parser.add_argument("--runs", type=int, default=3, help="每种缓存模式重复次数（默认 3）")
    parser.add_argument("--output", default="eval/results", help="结果输出目录（默认 eval/results）")
    args = parser.parse_args()

    if args.runs <= 0:
        raise SystemExit("错误：--runs 必须为正整数")

    pdfs = _collect_pdf_files(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载向量库与模型（首次约需 1–2 分钟）…")
    store = VisionVectorStore()

    results: List[Dict[str, Any]] = []
    print(f"开始上传链路 benchmark：{len(pdfs)} 个 PDF，mode={args.mode}，runs={args.runs}")
    for pdf_path in pdfs:
        for run_index in range(1, args.runs + 1):
            for cache_mode in _iter_modes(args.mode):
                result = _benchmark_single_run(store, pdf_path, cache_mode, run_index)
                print(
                    f"  [{cache_mode}] {pdf_path.name} run={run_index} "
                    f"pages={result['page_count']} total={result['total_index_ms']:.1f}ms "
                    f"render={result['pdf_render_ms']:.1f}ms embed={result['embedding_ms']:.1f}ms "
                    f"upsert={result['upsert_ms']:.1f}ms"
                )
                results.append(result)

    summary = {
        "_type": "summary",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmark": "upload",
        "config": {
            "input": str(Path(args.input).expanduser()),
            "mode": args.mode,
            "runs": args.runs,
            "file_count": len(pdfs),
        },
        "by_mode": _build_summary(results),
    }

    timestamp = summary["timestamp"]
    output_path = output_dir / f"upload_benchmark_{timestamp}.jsonl"
    rows = [summary] + results
    _write_jsonl(output_path, rows)

    print("\n上传链路汇总（按 cache_mode）")
    print("=" * 72)
    for mode, mode_summary in summary["by_mode"].items():
        print(f"[{mode}]")
        for field in ("pdf_render_ms", "embedding_ms", "upsert_ms", "total_index_ms", "pages_per_second"):
            stats = mode_summary.get(field, {})
            if stats:
                print(
                    f"  {field:<18} p50={stats['p50']:>8.1f}  "
                    f"p95={stats['p95']:>8.1f}  mean={stats['mean']:>8.1f}"
                )
        print()

    print(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()