#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_METRICS = [
    "sample_count",
    "raw_page_hit_at_1",
    "raw_page_hit_at_3",
    "raw_page_mrr",
    "final_page_hit_at_3",
    "final_page_mrr",
    "avg_colpali_query_embedding_ms",
    "avg_muvera_query_embedding_ms",
    "avg_total_retrieval_ms",
    "avg_client_total_chat_ms",
    "fallback_rate",
    "citation_has_gold_evidence_rate",
    "answer_contains_gold_answer_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-mode RAG eval summaries into a single retrieval ablation comparison summary.",
    )
    parser.add_argument(
        "--mode-summary",
        action="append",
        required=True,
        help="Mode/path pair such as two_stage=data/eval_runs/.../summary.json. Repeat once per mode.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="two_stage",
        help="Baseline mode used for delta calculations. Defaults to two_stage.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to write the comparison summary JSON. Defaults to comparison_summary.json next to the first input summary.",
    )
    return parser.parse_args()


def parse_mode_summary_arg(raw_value: str) -> Tuple[str, Path]:
    if "=" not in raw_value:
        raise SystemExit(f"Invalid --mode-summary value: {raw_value!r}. Expected mode=path.")
    mode, path_str = raw_value.split("=", 1)
    mode = mode.strip()
    if not mode:
        raise SystemExit(f"Invalid mode in --mode-summary value: {raw_value!r}")
    return mode, Path(path_str.strip())


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_output_path(explicit_output_file: str | None, first_summary_path: Path) -> Path:
    if explicit_output_file:
        return Path(explicit_output_file)
    return first_summary_path.parent / "comparison_summary.json"


def compute_delta(value: Any, baseline_value: Any) -> Any:
    if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
        return round(float(value) - float(baseline_value), 4)
    return None


def main() -> None:
    args = parse_args()
    mode_paths: List[Tuple[str, Path]] = [parse_mode_summary_arg(item) for item in args.mode_summary]

    seen_modes = set()
    mode_payloads: Dict[str, Dict[str, Any]] = {}
    for mode, path in mode_paths:
        if mode in seen_modes:
            raise SystemExit(f"Duplicate mode provided: {mode}")
        seen_modes.add(mode)
        summary_payload = load_summary(path)
        mode_payloads[mode] = {
            "summary_path": str(path),
            "metrics": {metric: summary_payload.get(metric) for metric in DEFAULT_METRICS},
        }

    baseline_mode = args.baseline_mode if args.baseline_mode in mode_payloads else next(iter(mode_payloads))
    baseline_metrics = mode_payloads[baseline_mode]["metrics"]

    deltas_vs_baseline: Dict[str, Dict[str, Any]] = {}
    for mode, payload in mode_payloads.items():
        if mode == baseline_mode:
            continue
        metrics = payload["metrics"]
        deltas_vs_baseline[mode] = {
            f"{metric}_delta": compute_delta(metrics.get(metric), baseline_metrics.get(metric))
            for metric in DEFAULT_METRICS
            if metric != "sample_count"
        }

    comparison_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_mode": baseline_mode,
        "modes": {
            mode: {
                "summary_path": payload["summary_path"],
                "metrics": payload["metrics"],
            }
            for mode, payload in mode_payloads.items()
        },
        "deltas_vs_baseline": deltas_vs_baseline,
    }

    output_path = build_output_path(args.output_file, mode_paths[0][1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"baseline_mode={baseline_mode}")
    for mode, payload in comparison_payload["modes"].items():
        metrics = payload["metrics"]
        print(
            f"- {mode}: raw_hit@3={metrics.get('raw_page_hit_at_3')} raw_mrr={metrics.get('raw_page_mrr')} "
            f"retrieval_ms={metrics.get('avg_total_retrieval_ms')}"
        )
    print(output_path)


if __name__ == "__main__":
    main()