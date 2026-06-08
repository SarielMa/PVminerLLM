#!/usr/bin/env python3
"""Split 70B PV lm-eval sample rows by source-message file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("data_from_sources")
DEFAULT_OUTPUT_DIR = Path(
    "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/source_splits"
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
}
FALLBACK_SOURCE = "YNHH"


def source_label(source_path: Path) -> str:
    name = source_path.name.lower()
    if name.startswith("bethesda"):
        return "Bethesda"
    if name.startswith("survey"):
        return "Survey"
    if name.startswith("woven"):
        return "Woven"
    return source_path.stem


def normalize_context(context: Any) -> str:
    if context is None:
        return ""
    return str(context).strip()


def parse_sample_arg(value: str) -> tuple[str, Path]:
    """Parse either PATH or LABEL=PATH."""
    if "=" in value:
        label, sample_path = value.split("=", 1)
        return label, Path(sample_path)

    sample_path = Path(value)
    return sample_path.stem, sample_path


def load_source_index(source_dir: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return normalized context -> source label/file maps."""
    context_to_source: dict[str, str] = {}
    context_to_files: dict[str, list[str]] = defaultdict(list)
    conflicts: dict[str, set[str]] = defaultdict(set)

    for source_path in sorted(source_dir.glob("*.jsonl")):
        label = source_label(source_path)
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                row = json.loads(line)
                key = normalize_context(row.get("context"))
                if not key:
                    continue

                if key in context_to_source and context_to_source[key] != label:
                    conflicts[key].update({context_to_source[key], label})
                    continue

                context_to_source[key] = label
                context_to_files[key].append(str(source_path))

    if conflicts:
        examples = "; ".join(
            f"{sorted(labels)}" for labels in list(conflicts.values())[:5]
        )
        raise ValueError(f"Contexts found in multiple source files: {examples}")

    return context_to_source, context_to_files


def split_sample(
    sample_label: str,
    sample_path: Path,
    output_dir: Path,
    context_to_source: dict[str, str],
    context_to_files: dict[str, list[str]],
) -> list[dict[str, Any]]:
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)
            # lm-eval stores the original message text here:
            # {"doc": {"query": ..., "answer": ..., "context": ...}}
            context = row.get("doc", {}).get("context", "")
            key = normalize_context(context)
            source = context_to_source.get(key, FALLBACK_SOURCE)
            source_files = context_to_files.get(key, [])

            split_row = dict(row)
            split_row["source_dataset"] = source
            split_row["source_files"] = source_files
            split_row["source_match"] = "doc.context.strip()"
            source_rows[source].append(split_row)

    sample_output_dir = output_dir / sample_label
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    ordered_sources = sorted(
        set(context_to_source.values()) | {FALLBACK_SOURCE}
    )
    for source in ordered_sources:
        rows = source_rows.get(source, [])
        output_path = sample_output_dir / f"{source}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        summary_rows.append(
            {
                "sample": sample_label,
                "source_dataset": source,
                "n_rows": len(rows),
                "output_path": str(output_path),
            }
        )

    return summary_rows


def write_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    headers = ["sample", "source_dataset", "n_rows"]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in summary_rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in summary_rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split 70B lm-eval sample rows into Bethesda, Survey, Woven, and "
            "YNHH buckets by matching doc.context to data_from_sources/*.jsonl."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "samples",
        nargs="*",
        help="Sample JSONL paths, optionally as LABEL=PATH. Defaults to 70B raw and SFT samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = dict(DEFAULT_SAMPLES)
    if args.samples:
        samples = dict(parse_sample_arg(sample) for sample in args.samples)

    context_to_source, context_to_files = load_source_index(args.source_dir)

    summary_rows = []
    for sample_label, sample_path in samples.items():
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        summary_rows.extend(
            split_sample(
                sample_label,
                sample_path,
                args.output_dir,
                context_to_source,
                context_to_files,
            )
        )

    summary_path = write_summary(summary_rows, args.output_dir)
    print_summary(summary_rows)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
