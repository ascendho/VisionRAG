"""
VisionRAG 检索质量评估器（带金标）

测量系统在带人工标注的 gold-page 查询集上的检索准确性。

主指标：
  Recall@1    top-1 结果中包含 gold page 的查询比例
  Recall@3    top-3 结果中包含 gold page 的查询比例
  Recall@5    top-5 结果中包含 gold page 的查询比例
  MRR         首个 gold page 命中的平均倒数排名（越高越好，1 = 第 1 位命中）
  NDCG@5      折损累积增益@5，二值相关度（命中 gold = 1，未命中 = 0）
  doc_routing 文档路由准确率：top-k 结果中是否包含来自正确文档的页面

输入格式（JSONL，每行一条）：
  {
    "query_id":        "q001",
    "query_text":      "...",
    "document_name":   "Report.pdf",      # 与系统 document_name 精确一致
    "gold_page_numbers": [3, 4],           # 1-indexed
    "primary_answer":  "...",              # 人工参考答案（供 answer_eval 使用）
    "query_type":      "factual",
    "difficulty":      "easy"
  }

用法：
  python -m eval.retrieval_eval \\
      --queries eval/queries/gold_queries.jsonl \\
      --top-k 5 --prefetch-multiplier 10
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VisionVectorStore


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _load_gold_queries(query_path: str) -> List[Dict[str, Any]]:
    path = Path(query_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"查询文件不存在：{query_path}")
    queries = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {lineno} 行 JSON 解析失败：{exc}") from exc
            for required in ("query_text", "document_name", "gold_page_numbers"):
                if required not in row:
                    raise ValueError(f"第 {lineno} 行缺少必填字段：'{required}'")
            if "query_id" not in row:
                row["query_id"] = f"q{lineno:03d}"
            if not isinstance(row["gold_page_numbers"], list) or not row["gold_page_numbers"]:
                raise ValueError(
                    f"第 {lineno} 行 'gold_page_numbers' 必须是非空整数列表"
                )
            queries.append(row)
    if not queries:
        raise ValueError("查询文件为空，至少需要一条标注查询。")
    return queries


def _resolve_document_ids(
    store: VisionVectorStore, document_name: str
) -> List[str]:
    docs = store.get_all_documents()
    matched = [d["document_id"] for d in docs if d.get("document_name") == document_name]
    if not matched:
        available = sorted({d.get("document_name", "") for d in docs})
        raise ValueError(
            f"找不到文档 '{document_name}'。\n"
            f"当前已索引文档（{len(available)} 份）：\n"
            + "\n".join(f"  - {n}" for n in available)
        )
    return matched


# ── 单项指标计算 ──────────────────────────────────────────────────────────────

def _recall_at_k(retrieved: List[int], gold: List[int], k: int) -> float:
    """1 如果 top-k 结果中有任意 gold page，否则 0。"""
    gold_set = set(gold)
    return 1.0 if any(p in gold_set for p in retrieved[:k]) else 0.0


def _mrr(retrieved: List[int], gold: List[int]) -> float:
    """第一个 gold page 命中的倒数排名；未命中返回 0。"""
    gold_set = set(gold)
    for rank, page in enumerate(retrieved, 1):
        if page in gold_set:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved: List[int], gold: List[int], k: int) -> float:
    """二值相关度 NDCG@k（命中 gold = 1，未命中 = 0）。"""
    gold_set = set(gold)
    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(retrieved[:k], 1)
        if page in gold_set
    )
    # IDCG：最理想情况下 min(|gold|, k) 个命中全部排在最前面
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


# ── 单条查询评估 ──────────────────────────────────────────────────────────────

def evaluate_single_query(
    store: VisionVectorStore,
    query: Dict[str, Any],
    top_k: int,
    prefetch_multiplier: int,
) -> Dict[str, Any]:
    """对单条标注查询执行检索并返回指标字典。"""
    document_ids = _resolve_document_ids(store, query["document_name"])
    gold_pages = [int(p) for p in query["gold_page_numbers"]]

    results = store.retrieve_with_two_stage(
        query_text=query["query_text"],
        top_k=top_k,
        prefetch_multiplier=prefetch_multiplier,
        document_ids=document_ids,
    )

    retrieved_pages = [int(r["page_number"]) for r in results]
    retrieved_doc_ids = {r["document_id"] for r in results}
    gold_doc_ids = set(document_ids)

    scores = [float(r["score"]) for r in results]
    top_score = scores[0] if scores else 0.0

    return {
        "query_id": query.get("query_id", ""),
        "query_text": query["query_text"],
        "document_name": query["document_name"],
        "query_type": query.get("query_type", ""),
        "difficulty": query.get("difficulty", ""),
        "gold_page_numbers": gold_pages,
        "retrieved_page_numbers": retrieved_pages,
        "retrieved_scores": [round(s, 4) for s in scores],
        "top_score": round(top_score, 4),
        "recall_at_1": _recall_at_k(retrieved_pages, gold_pages, 1),
        "recall_at_3": _recall_at_k(retrieved_pages, gold_pages, 3),
        "recall_at_5": _recall_at_k(retrieved_pages, gold_pages, 5),
        "mrr": round(_mrr(retrieved_pages, gold_pages), 4),
        "ndcg_at_5": _ndcg_at_k(retrieved_pages, gold_pages, 5),
        "doc_routing_correct": int(bool(retrieved_doc_ids & gold_doc_ids)),
    }


# ── 聚合与汇报 ────────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总所有查询的平均指标，并按 query_type 和 difficulty 分层。"""
    metric_keys = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5", "doc_routing_correct")

    overall: Dict[str, float] = {
        k: _mean([r[k] for r in rows]) for k in metric_keys
    }

    # 按 query_type 分层
    by_type: Dict[str, Dict[str, float]] = {}
    for qtype in sorted({r["query_type"] for r in rows if r["query_type"]}):
        sub = [r for r in rows if r["query_type"] == qtype]
        by_type[qtype] = {k: _mean([r[k] for r in sub]) for k in metric_keys}
        by_type[qtype]["count"] = len(sub)

    # 按 difficulty 分层
    by_difficulty: Dict[str, Dict[str, float]] = {}
    for diff in sorted({r["difficulty"] for r in rows if r["difficulty"]}):
        sub = [r for r in rows if r["difficulty"] == diff]
        by_difficulty[diff] = {k: _mean([r[k] for r in sub]) for k in metric_keys}
        by_difficulty[diff]["count"] = len(sub)

    return {
        "overall": overall,
        "by_query_type": by_type,
        "by_difficulty": by_difficulty,
        "total_queries": len(rows),
    }


def _print_report(agg: Dict[str, Any], config: Dict[str, Any]) -> None:
    print("\n检索质量评估报告")
    print("=" * 68)
    print(f"  查询总数：{agg['total_queries']}  top_k={config['top_k']}  "
          f"prefetch_multiplier={config['prefetch_multiplier']}")
    print()

    ov = agg["overall"]
    print("【总体指标】")
    print(f"  Recall@1  = {ov['recall_at_1']:.4f}")
    print(f"  Recall@3  = {ov['recall_at_3']:.4f}")
    print(f"  Recall@5  = {ov['recall_at_5']:.4f}")
    print(f"  MRR       = {ov['mrr']:.4f}")
    print(f"  NDCG@5    = {ov['ndcg_at_5']:.4f}")
    print(f"  文档路由  = {ov['doc_routing_correct']:.4f}")

    if agg["by_query_type"]:
        print("\n【按 query_type 分层】")
        print(f"  {'类型':<14} {'n':>4}  {'R@1':>6}  {'R@3':>6}  {'R@5':>6}  {'MRR':>6}  {'NDCG5':>7}")
        print("  " + "-" * 58)
        for qtype, m in agg["by_query_type"].items():
            print(
                f"  {qtype:<14} {int(m['count']):>4}  "
                f"{m['recall_at_1']:>6.4f}  {m['recall_at_3']:>6.4f}  "
                f"{m['recall_at_5']:>6.4f}  {m['mrr']:>6.4f}  {m['ndcg_at_5']:>7.4f}"
            )

    if agg["by_difficulty"]:
        print("\n【按难度分层】")
        print(f"  {'难度':<8} {'n':>4}  {'R@1':>6}  {'R@3':>6}  {'R@5':>6}  {'MRR':>6}  {'NDCG5':>7}")
        print("  " + "-" * 50)
        for diff, m in agg["by_difficulty"].items():
            print(
                f"  {diff:<8} {int(m['count']):>4}  "
                f"{m['recall_at_1']:>6.4f}  {m['recall_at_3']:>6.4f}  "
                f"{m['recall_at_5']:>6.4f}  {m['mrr']:>6.4f}  {m['ndcg_at_5']:>7.4f}"
            )


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_retrieval_eval(
    store: VisionVectorStore,
    queries: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    prefetch_multiplier: int = 10,
    output_dir: Path,
    label: str = "baseline",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对整个查询集执行检索评估，返回 (per_query_rows, aggregate) 元组。

    对外暴露为函数，便于消融实验脚本直接调用。
    """
    rows: List[Dict[str, Any]] = []
    for query in queries:
        row = evaluate_single_query(
            store=store,
            query=query,
            top_k=top_k,
            prefetch_multiplier=prefetch_multiplier,
        )
        gold_hit = "✓" if row["recall_at_1"] == 1.0 else "✗"
        print(
            f"  [{gold_hit}] {row['query_id']} "
            f"R@1={row['recall_at_1']:.1f} R@3={row['recall_at_3']:.1f} "
            f"R@5={row['recall_at_5']:.1f} MRR={row['mrr']:.3f} "
            f"top_score={row['top_score']:.3f}"
        )
        rows.append(row)

    agg = _aggregate(rows)
    return rows, agg


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionRAG 检索质量评估（带 gold-page）")
    parser.add_argument("--queries", required=True, help="金标查询文件路径（JSONL）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回最多页数（默认 5）")
    parser.add_argument(
        "--prefetch-multiplier", type=int, default=10,
        help="MUVERA 海选倍率（默认 10）"
    )
    parser.add_argument("--output", default="eval/results", help="结果输出目录（默认 eval/results）")
    parser.add_argument("--label", default="baseline", help="实验标签，写入结果文件（默认 baseline）")
    args = parser.parse_args()

    queries = _load_gold_queries(args.queries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载向量库与模型（首次约需 1–2 分钟）…")
    store = VisionVectorStore()

    print(
        f"开始检索评估：{len(queries)} 条查询，"
        f"top_k={args.top_k}，prefetch_multiplier={args.prefetch_multiplier}"
    )
    rows, agg = run_retrieval_eval(
        store=store,
        queries=queries,
        top_k=args.top_k,
        prefetch_multiplier=args.prefetch_multiplier,
        output_dir=output_dir,
        label=args.label,
    )

    config = {
        "queries": str(Path(args.queries).expanduser()),
        "top_k": args.top_k,
        "prefetch_multiplier": args.prefetch_multiplier,
        "label": args.label,
        "query_count": len(queries),
    }
    _print_report(agg, config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_row = {
        "_type": "summary",
        "timestamp": timestamp,
        "benchmark": "retrieval",
        "label": args.label,
        "config": config,
        "aggregate": agg,
    }

    output_path = output_dir / f"retrieval_eval_{args.label}_{timestamp}.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_row, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n结果已保存：{output_path}")


if __name__ == "__main__":
    main()
