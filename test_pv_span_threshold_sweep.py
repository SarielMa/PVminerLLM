#!/usr/bin/env python3
"""Recompute PV extraction metrics while sweeping the span threshold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pv_utils import (
    calculate_code,
    calculate_subcode,
    relaxed_match_evaluation_with_full_containment,
    safe_json_loads,
)


DEFAULT_SAMPLES = {
    "raw_2shot_70b": Path(
        "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "meta-llama__Llama-3.3-70B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T15-26-00.777869.jsonl"
    ),
    "sft_0shot_70b": Path(
        "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__llama3.3_70b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T18-14-33.129764.jsonl"
    ),
    "raw_2shot_1.5b": Path(
        "runs_pv_epoch10_b200/qwen2.5_1.5b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "Qwen__Qwen2.5-1.5B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T19-11-02.330653.jsonl"
    ),
    "sft_0shot_1.5b": Path(
        "runs_pv_epoch10_b200/qwen2.5_1.5b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__qwen2.5_1.5b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T19-30-10.674621.jsonl"
    ),
}

DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(60, 100, 5)]


def parse_sample_arg(value: str) -> tuple[str, Path]:
    """Parse either PATH or LABEL=PATH."""
    if "=" in value:
        label, sample_path = value.split("=", 1)
        return label, Path(sample_path)

    sample_path = Path(value)
    return infer_label(sample_path), sample_path


def infer_label(sample_path: Path) -> str:
    parts = set(sample_path.parts)
    if "raw_2shot_lm_eval_results" in parts:
        return "raw_2shot_70b"
    if "sft_lm_eval_results" in parts:
        return "sft_0shot_70b"
    return sample_path.stem


def load_eval_pairs(sample_path: Path) -> list[tuple[str, str]]:
    """Load (ground_truth, prediction) pairs from lm-eval sample JSONL."""
    pairs: list[tuple[str, str]] = []

    with sample_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            metric_pair = row.get("evaluate_eppc")
            if isinstance(metric_pair, list) and len(metric_pair) >= 2:
                pairs.append((metric_pair[0], metric_pair[1]))
                continue

            target = row.get("target")
            filtered_resps = row.get("filtered_resps")
            if isinstance(target, str) and isinstance(filtered_resps, list) and filtered_resps:
                pairs.append((target, filtered_resps[0]))
                continue

            raise ValueError(
                f"{sample_path}:{line_number} does not contain evaluate_eppc "
                "or target/filtered_resps fields"
            )

    return pairs


def get_results(value: Any) -> list[dict[str, Any]]:
    parsed = safe_json_loads(value)
    if not isinstance(parsed, dict):
        return []

    results = parsed.get("results") or []
    if not isinstance(results, list):
        return []

    return [item for item in results if isinstance(item, dict)]


def split_eval_fields(
    pairs: list[tuple[str, str]],
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]]]:
    true_codes: list[list[Any]] = []
    pred_codes: list[list[Any]] = []
    true_subcodes: list[list[Any]] = []
    pred_subcodes: list[list[Any]] = []
    true_spans: list[list[Any]] = []
    pred_spans: list[list[Any]] = []

    for ground_truth, prediction in pairs:
        true_results = get_results(ground_truth)
        pred_results = get_results(prediction)

        true_codes.append([item.get("Code") for item in true_results])
        pred_codes.append([item.get("Code") for item in pred_results])
        true_subcodes.append([item.get("Sub-code") for item in true_results])
        pred_subcodes.append([item.get("Sub-code") for item in pred_results])
        true_spans.append([item.get("Span") for item in true_results])
        pred_spans.append([item.get("Span") for item in pred_results])

    return true_codes, pred_codes, true_subcodes, pred_subcodes, true_spans, pred_spans


def round_metric(value: float) -> float:
    return round(value, 4)


def evaluate_sample(
    label: str,
    sample_path: Path,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    pairs = load_eval_pairs(sample_path)
    (
        true_codes,
        pred_codes,
        true_subcodes,
        pred_subcodes,
        true_spans,
        pred_spans,
    ) = split_eval_fields(pairs)

    code_p, code_r, code_f1 = calculate_code(true_codes, pred_codes)
    subcode_p, subcode_r, subcode_f1 = calculate_subcode(true_subcodes, pred_subcodes)

    rows = []
    for threshold in thresholds:
        span_p, span_r, span_f1 = relaxed_match_evaluation_with_full_containment(
            true_spans,
            pred_spans,
            jaccard_threshold=threshold,
        )
        rows.append(
            {
                "label": label,
                "sample_path": str(sample_path),
                "n_items": len(pairs),
                "threshold": f"{threshold:.2f}",
                "code_P": round_metric(code_p),
                "code_R": round_metric(code_r),
                "code_f1": round_metric(code_f1),
                "subcode_P": round_metric(subcode_p),
                "subcode_R": round_metric(subcode_r),
                "subcode_f1": round_metric(subcode_f1),
                "span_P": round_metric(span_p),
                "span_R": round_metric(span_r),
                "span_f1": round_metric(span_f1),
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def print_rows(rows: list[dict[str, Any]]) -> None:
    headers = [
        "label",
        "n_items",
        "threshold",
        "span_P",
        "span_R",
        "span_f1",
        "code_f1",
        "subcode_f1",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep PV span relaxed-match thresholds using lm-eval sample JSONL "
            "files. Defaults to both 70B sample files in runs_pv_epoch10_b200."
        )
    )
    parser.add_argument(
        "samples",
        nargs="*",
        help="Sample JSONL paths, optionally as LABEL=PATH. Defaults to the two 70B sample files.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Span Jaccard thresholds to evaluate.",
    )
    parser.add_argument("--csv-out", type=Path, help="Optional CSV output path.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = dict(DEFAULT_SAMPLES)
    if args.samples:
        samples = dict(parse_sample_arg(sample) for sample in args.samples)

    rows: list[dict[str, Any]] = []
    for label, sample_path in samples.items():
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        rows.extend(evaluate_sample(label, sample_path, args.thresholds))

    print_rows(rows)

    if args.csv_out:
        write_csv(rows, args.csv_out)
    if args.json_out:
        write_json(rows, args.json_out)


if __name__ == "__main__":
    main()
