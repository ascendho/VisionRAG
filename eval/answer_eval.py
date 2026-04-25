"""
VisionRAG 回答有据可依评估器（带金标）

在检索评估的基础上进一步验证：
  1. 系统检索到的证据页是否与 gold page 重叠（grounding）
  2. 基于检索证据页生成的回答文本（供人工核对）

这一评估器**不**自动打分回答质量，因为在轻量标注预算下，
自动化 NLP 指标（ROUGE、BERTScore）不如人工核对来得可靠。
生成的答案和金标参考答案会并排输出，便于快速人工审阅。

用法（在 retrieval_eval 跑完之后使用，或直接从 gold 文件驱动）：
  python -m eval.answer_eval \\
      --queries eval/queries/gold_queries.jsonl \\
      --top-k 5 --min-score 0.5 --max-tokens 300

输入格式（与 retrieval_eval 相同的 gold JSONL）：
  每行需含 query_text、document_name、gold_page_numbers、primary_answer
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_generator import generate_answer_stream
from src.vector_store import VisionVectorStore
from eval.retrieval_eval import (
    _load_gold_queries,
    _resolve_document_ids,
    evaluate_single_query,
)


# ── 核心评估逻辑 ──────────────────────────────────────────────────────────────

def _is_grounded(
    retrieved_pages: List[int],
    gold_pages: List[int],
    accepted_pages: List[int],
) -> bool:
    """
    判断生成时使用的证据页（accepted_pages）是否与 gold 页重叠。
    若 accepted_pages 为空（被 min_score 全部过滤），则降回检索页判断。
    """
    evidence_pages = accepted_pages if accepted_pages else retrieved_pages
    gold_set = set(gold_pages)
    return any(p in gold_set for p in evidence_pages)


def evaluate_single_answer(
    store: VisionVectorStore,
    query: Dict[str, Any],
    top_k: int,
    prefetch_multiplier: int,
    min_score: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """对单条查询执行检索 + 生成，返回评估结果字典。"""
    # 步骤 1：检索（复用 retrieval_eval 的逻辑）
    retrieval = evaluate_single_query(
        store=store,
        query=query,
        top_k=top_k,
        prefetch_multiplier=prefetch_multiplier,
    )

    # 步骤 2：按 min_score 过滤，与生产路由行为一致
    results_full = store.retrieve_with_two_stage(
        query_text=query["query_text"],
        top_k=top_k,
        prefetch_multiplier=prefetch_multiplier,
        document_ids=_resolve_document_ids(store, query["document_name"]),
    )
    accepted = [r for r in results_full if r["score"] >= min_score]
    if not accepted and results_full:
        accepted = results_full[:1]  # 至少保留最高分，与生产路由一致
    accepted_pages = [int(r["page_number"]) for r in accepted]
    evidence_paths = [r["image_path"] for r in accepted if r.get("image_path")]

    # 步骤 3：生成回答
    answer_tokens: List[str] = []
    if evidence_paths:
        for token in generate_answer_stream(
            query_text=query["query_text"],
            image_paths=evidence_paths,
            max_tokens=max_tokens,
        ):
            answer_tokens.append(token)
    generated_answer = "".join(answer_tokens).strip()

    # 步骤 4：判断 grounding
    grounded = _is_grounded(
        retrieved_pages=retrieval["retrieved_page_numbers"],
        gold_pages=retrieval["gold_page_numbers"],
        accepted_pages=accepted_pages,
    )

    return {
        "query_id": retrieval["query_id"],
        "query_text": retrieval["query_text"],
        "document_name": retrieval["document_name"],
        "query_type": retrieval.get("query_type", ""),
        "difficulty": retrieval.get("difficulty", ""),
        "gold_page_numbers": retrieval["gold_page_numbers"],
        "retrieved_page_numbers": retrieval["retrieved_page_numbers"],
        "accepted_page_numbers": accepted_pages,
        "top_score": retrieval["top_score"],
        # retrieval 指标（顺带保留，方便合并分析）
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_3": retrieval["recall_at_3"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr": retrieval["mrr"],
        "ndcg_at_5": retrieval["ndcg_at_5"],
        # grounding
        "grounded": int(grounded),
        "accepted_count": len(accepted),
        # 文本对（供人工核对）
        "generated_answer": generated_answer,
        "reference_answer": query.get("primary_answer", ""),
    }


# ── 人工核对报告 ──────────────────────────────────────────────────────────────

def _print_answer_report(rows: List[Dict[str, Any]]) -> None:
    grounded_count = sum(r["grounded"] for r in rows)
    grounding_rate = grounded_count / len(rows) if rows else 0.0

    print("\n回答有据可依评估报告")
    print("=" * 68)
    print(f"  查询总数：{len(rows)}")
    print(f"  有据可依率（grounding）：{grounded_count}/{len(rows)} = {grounding_rate:.2%}")
    print()
    print("── 逐条答案对比（供人工核对）─────────────────────────────────────")
    for row in rows:
        grounded_mark = "✓" if row["grounded"] else "✗"
        print(f"\n[{grounded_mark}] {row['query_id']} ({row['query_type']}, {row['difficulty']})")
        print(f"  问题      : {row['query_text']}")
        print(f"  gold 页   : {row['gold_page_numbers']}")
        print(f"  检索页    : {row['retrieved_page_numbers']}")
        print(f"  证据页    : {row['accepted_page_numbers']}")
        print(f"  生成回答  : {row['generated_answer'][:200]}{'…' if len(row['generated_answer']) > 200 else ''}")
        print(f"  参考答案  : {row['reference_answer']}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionRAG 回答有据可依评估（带 gold-page，供人工核对）"
    )
    parser.add_argument("--queries", required=True, help="金标查询文件路径（JSONL）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回最多页数（默认 5）")
    parser.add_argument(
        "--prefetch-multiplier", type=int, default=10,
        help="MUVERA 海选倍率（默认 10）"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.5,
        help="证据接受最低分（默认 0.5；与生产路由保持一致）"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="生成最大 token 数（默认 300，评估时建议设短一些以节省时间）"
    )
    parser.add_argument("--output", default="eval/results", help="结果输出目录（默认 eval/results）")
    parser.add_argument("--label", default="baseline", help="实验标签（默认 baseline）")
    args = parser.parse_args()

    queries = _load_gold_queries(args.queries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载向量库与模型（首次约需 1–2 分钟）…")
    store = VisionVectorStore()

    print(
        f"开始回答评估：{len(queries)} 条查询，"
        f"top_k={args.top_k}，min_score={args.min_score}，max_tokens={args.max_tokens}"
    )
    rows: List[Dict[str, Any]] = []
    for query in queries:
        row = evaluate_single_answer(
            store=store,
            query=query,
            top_k=args.top_k,
            prefetch_multiplier=args.prefetch_multiplier,
            min_score=args.min_score,
            max_tokens=args.max_tokens,
        )
        grounded_mark = "✓" if row["grounded"] else "✗"
        print(
            f"  [{grounded_mark}] {row['query_id']} "
            f"grounded={row['grounded']} accepted={row['accepted_count']} "
            f"R@1={row['recall_at_1']:.1f} MRR={row['mrr']:.3f}"
        )
        rows.append(row)

    _print_answer_report(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_row = {
        "_type": "summary",
        "timestamp": timestamp,
        "benchmark": "answer_grounding",
        "label": args.label,
        "config": {
            "queries": str(Path(args.queries).expanduser()),
            "top_k": args.top_k,
            "prefetch_multiplier": args.prefetch_multiplier,
            "min_score": args.min_score,
            "max_tokens": args.max_tokens,
            "query_count": len(queries),
        },
        "grounding_rate": sum(r["grounded"] for r in rows) / len(rows) if rows else 0.0,
    }

    output_path = output_dir / f"answer_eval_{args.label}_{timestamp}.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_row, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n结果已保存：{output_path}")
    print("（请人工核对上方「生成回答 vs 参考答案」逐条比对，标注正确/错误）")


if __name__ == "__main__":
    main()
