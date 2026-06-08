#!/usr/bin/env python3
"""Report how often predicted PV spans are exact substrings of the input."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pv_utils import safe_json_loads


DEFAULT_SAMPLES = {
    "Two-shot 70B": Path(
        "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "meta-llama__Llama-3.3-70B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T15-26-00.777869.jsonl"
    ),
    "SFT 70B": Path(
        "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__llama3.3_70b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T18-14-33.129764.jsonl"
    ),
    "Two-shot 8B": Path(
        "runs_pv_epoch10_b200/llama3.1_8b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "meta-llama__Llama-3.1-8B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T18-16-49.757009.jsonl"
    ),
    "SFT 8B": Path(
        "runs_pv_epoch10_b200/llama3.1_8b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__llama3.1_8b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T18-47-41.195465.jsonl"
    ),
    "Two-shot 3B": Path(
        "runs_pv_epoch10_b200/llama3.2_3b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "meta-llama__Llama-3.2-3B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T18-49-04.277522.jsonl"
    ),
    "SFT 3B": Path(
        "runs_pv_epoch10_b200/llama3.2_3b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__llama3.2_3b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T19-09-33.716122.jsonl"
    ),
    "Two-shot 1.5B": Path(
        "runs_pv_epoch10_b200/qwen2.5_1.5b_instruct/sft_10ep/"
        "raw_2shot_lm_eval_results/PvExtraction_full/"
        "Qwen__Qwen2.5-1.5B-Instruct/"
        "samples_PvExtraction_full_2026-06-05T19-11-02.330653.jsonl"
    ),
    "SFT 1.5B": Path(
        "runs_pv_epoch10_b200/qwen2.5_1.5b_instruct/sft_10ep/"
        "sft_lm_eval_results/PvExtraction_full/"
        "__nfs__roberts__project__pi_sjf37__lm2445__PV_multiagent__sft_open__"
        "runs_pv_epoch10_b200__qwen2.5_1.5b_instruct__sft_10ep__merged/"
        "samples_PvExtraction_full_2026-06-05T19-30-10.674621.jsonl"
    ),
}


def parse_sample_arg(value: str) -> tuple[str, Path]:
    """Parse either PATH or LABEL=PATH."""
    if "=" in value:
        label, sample_path = value.split("=", 1)
        return label, Path(sample_path)
    sample_path = Path(value)
    return sample_path.stem, sample_path


def get_prediction(row: dict[str, Any]) -> str:
    metric_pair = row.get("evaluate_eppc")
    if isinstance(metric_pair, list) and len(metric_pair) >= 2:
        return metric_pair[1]

    filtered_resps = row.get("filtered_resps")
    if isinstance(filtered_resps, list) and filtered_resps:
        return filtered_resps[0]

    return ""


def get_predicted_spans(prediction: str) -> list[str]:
    parsed = safe_json_loads(prediction)
    if not isinstance(parsed, dict):
        return []

    results = parsed.get("results") or []
    if not isinstance(results, list):
        return []

    spans = []
    for item in results:
        if not isinstance(item, dict):
            continue
        span = item.get("Span")
        if isinstance(span, str):
            spans.append(span)

    return spans


def evaluate_sample(label: str, sample_path: Path) -> dict[str, Any]:
    exact_count = 0
    predicted_count = 0
    empty_span_count = 0
    parseable_rows = 0
    total_rows = 0

    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            total_rows += 1
            row = json.loads(line)
            context = row.get("doc", {}).get("context", "")
            prediction = get_prediction(row)
            spans = get_predicted_spans(prediction)
            if spans or prediction:
                parseable_rows += 1

            for span in spans:
                predicted_count += 1
                if span == "":
                    empty_span_count += 1
                if span in context:
                    exact_count += 1

    exact_pct = 100 * exact_count / predicted_count if predicted_count else 0.0
    hallucination_pct = (
        100 * (predicted_count - exact_count) / predicted_count
        if predicted_count
        else 0.0
    )
    return {
        "model": label,
        "sample_path": str(sample_path),
        "n_items": total_rows,
        "parseable_rows": parseable_rows,
        "predicted_spans": predicted_count,
        "exact_substring_spans": exact_count,
        "exact_substring_pct": round(exact_pct, 2),
        "non_substring_spans": predicted_count - exact_count,
        "hallucination_pct": round(hallucination_pct, 2),
        "empty_spans": empty_span_count,
    }


def print_rows(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "predicted_spans",
        "exact_substring_spans",
        "exact_substring_pct",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the percentage of predicted PV spans that are exact "
            "substrings of the input message."
        )
    )
    parser.add_argument(
        "samples",
        nargs="*",
        help="Sample JSONL paths, optionally as LABEL=PATH. Defaults to 70B and 1.5B files.",
    )
    parser.add_argument("--csv-out", type=Path, help="Optional CSV output path.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = dict(DEFAULT_SAMPLES)
    if args.samples:
        samples = dict(parse_sample_arg(sample) for sample in args.samples)

    rows = []
    for label, sample_path in samples.items():
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        rows.append(evaluate_sample(label, sample_path))

    print_rows(rows)

    if args.csv_out:
        write_csv(rows, args.csv_out)
    if args.json_out:
        write_json(rows, args.json_out)


if __name__ == "__main__":
    main()
