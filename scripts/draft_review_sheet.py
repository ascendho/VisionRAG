#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


MANUAL_FIELDNAMES = [
    "manual_answer_accuracy",
    "manual_faithfulness",
    "manual_citation_correctness",
    "reviewer_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a first-pass reviewed draft CSV from an unreviewed RAG eval review sheet.",
    )
    parser.add_argument("review_csv", help="Path to the unreviewed review_sheet.csv file.")
    parser.add_argument(
        "--output-csv",
        help="Path to write the reviewed draft CSV. Defaults to <input>_reviewed_draft.csv.",
    )
    return parser.parse_args()


def parse_bool_flag(value: str) -> bool:
    return str(value).strip() in {"1", "1.0", "true", "True"}


def trim_answer(answer_text: str) -> str:
    text = " ".join(str(answer_text or "").split())
    return text[:120]


def base_scores(row: Dict[str, str]) -> Tuple[float, float, float, str]:
    answer_matches = parse_bool_flag(row.get("answer_contains_gold_answer", ""))
    citation_matches = parse_bool_flag(row.get("citation_has_gold_evidence", ""))
    has_citation = parse_bool_flag(row.get("answer_has_citation", ""))
    final_hit = parse_bool_flag(row.get("final_page_hit_at_3", ""))
    fallback_used = parse_bool_flag(row.get("fallback_evidence_used", ""))

    if answer_matches and citation_matches:
        note = "答案与 gold 一致；引用命中 gold 页。"
        if fallback_used:
            note = "答案与 gold 一致；虽然走了 fallback，但最终引用仍命中 gold 页。"
        return 1.0, 1.0, 1.0, note

    if answer_matches and has_citation and not citation_matches:
        faithfulness = 1.0 if final_hit else 0.5
        note = "答案看起来正确，但引用未命中 benchmark 标注的 gold 页。"
        if not final_hit:
            note = "答案看起来正确，但 final evidence 未命中 gold 页；需再核证据是否等价。"
        return 1.0, faithfulness, 0.5, note

    if not answer_matches and citation_matches:
        note = "答案与 gold 表述不完全一致，但引用命中 gold 页。"
        return 0.5, 1.0, 1.0, note

    note = "答案或证据存在明显不确定性，需重点人工确认。"
    faithfulness = 0.5 if has_citation else 0.0
    citation_score = 0.5 if has_citation else 0.0
    return 0.0, faithfulness, citation_score, note


ROW_OVERRIDES: Dict[str, Tuple[float, float, float, str]] = {
    "auto-006": (
        1.0,
        1.0,
        0.5,
        "回答给出了正确上限 128k，但引用页不是 benchmark 标注的 gold 页。",
    ),
    "auto-014": (
        1.0,
        1.0,
        1.0,
        "答案采用释义而非逐字复述 gold，但语义与定义页一致。",
    ),
    "auto-015": (
        1.0,
        1.0,
        1.0,
        "答案把第4层职责说全了；虽未逐字匹配 gold，但与传输层定义页一致。",
    ),
    "auto-018": (
        1.0,
        0.5,
        0.5,
        "答案与 gold 一致，但引用未命中 benchmark 标注的 gold 页；返回证据是否等价仍需人工再核。",
    ),
    "auto-020": (
        1.0,
        1.0,
        1.0,
        "答案把英文 gold 翻成中文表达，语义与总结页一致。",
    ),
}


def format_score(value: float) -> str:
    if value in {0.0, 1.0}:
        return str(int(value))
    return str(value)


def default_output_path(review_csv: Path) -> Path:
    if review_csv.name.endswith(".csv"):
        stem = review_csv.name[:-4]
        return review_csv.with_name(f"{stem}_reviewed_draft.csv")
    return review_csv.with_name(f"{review_csv.name}_reviewed_draft.csv")


def generate_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    drafted_rows: List[Dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        benchmark_id = updated.get("benchmark_id", "")
        scores = ROW_OVERRIDES.get(benchmark_id, base_scores(updated))
        answer_score, faithfulness_score, citation_score, note = scores

        updated["manual_answer_accuracy"] = format_score(answer_score)
        updated["manual_faithfulness"] = format_score(faithfulness_score)
        updated["manual_citation_correctness"] = format_score(citation_score)
        updated["reviewer_notes"] = note
        drafted_rows.append(updated)
    return drafted_rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.review_csv)
    output_path = Path(args.output_csv) if args.output_csv else default_output_path(input_path)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise SystemExit("review CSV is empty")
        fieldnames = list(reader.fieldnames or [])

    missing_fields = [field for field in MANUAL_FIELDNAMES if field not in fieldnames]
    if missing_fields:
        raise SystemExit(f"review CSV is missing fields: {', '.join(missing_fields)}")

    drafted_rows = generate_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(drafted_rows)

    print(output_path)


if __name__ == "__main__":
    main()