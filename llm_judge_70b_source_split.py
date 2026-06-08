#!/usr/bin/env python3
"""Split 70B PV result rows by source using an LLM judge.

This script sends message contexts to an OpenAI-compatible chat completion API.
Use a local endpoint for PHI-sensitive data, or confirm that your external API
use is approved before running it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_SOURCE_DIR = Path("data_from_sources")
DEFAULT_OUTPUT_DIR = Path(
    "runs_pv_epoch10_b200/llama3.3_70b_instruct/sft_10ep/source_splits_llm_judge"
)
DEFAULT_JUDGMENTS = DEFAULT_OUTPUT_DIR / "source_judgments.jsonl"
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
LABELS = ("Bethesda", "Survey", "Woven", "YNHH")


def normalize_context(context: Any) -> str:
    if context is None:
        return ""
    return str(context).strip()


def source_label(source_path: Path) -> str:
    name = source_path.name.lower()
    if name.startswith("bethesda"):
        return "Bethesda"
    if name.startswith("survey"):
        return "Survey"
    if name.startswith("woven"):
        return "Woven"
    return source_path.stem


def parse_sample_arg(value: str) -> tuple[str, Path]:
    """Parse either PATH or LABEL=PATH."""
    if "=" in value:
        label, sample_path = value.split("=", 1)
        return label, Path(sample_path)

    sample_path = Path(value)
    return sample_path.stem, sample_path


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + " ... [truncated]"


def load_source_contexts(source_dir: Path) -> dict[str, list[str]]:
    contexts_by_source: dict[str, list[str]] = {label: [] for label in LABELS[:-1]}
    seen_by_source: dict[str, set[str]] = {label: set() for label in LABELS[:-1]}

    for source_path in sorted(source_dir.glob("*.jsonl")):
        label = source_label(source_path)
        if label not in contexts_by_source:
            continue

        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)
                context = normalize_context(row.get("context"))
                if context and context not in seen_by_source[label]:
                    contexts_by_source[label].append(context)
                    seen_by_source[label].add(context)

    return contexts_by_source


def select_examples(contexts: list[str], k: int) -> list[str]:
    if k <= 0 or len(contexts) <= k:
        return contexts[:]

    if k == 1:
        return [contexts[0]]

    positions = [round(i * (len(contexts) - 1) / (k - 1)) for i in range(k)]
    return [contexts[position] for position in positions]


def build_source_examples(
    contexts_by_source: dict[str, list[str]],
    examples_per_source: int,
    example_max_chars: int,
) -> str:
    blocks = []
    for label in LABELS[:-1]:
        examples = select_examples(contexts_by_source[label], examples_per_source)
        lines = [f"{label} examples:"]
        for i, context in enumerate(examples, start=1):
            lines.append(f"{i}. {truncate_text(context, example_max_chars)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def load_sample_rows(samples: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for label, sample_path in samples.items():
        rows = []
        with sample_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        rows_by_sample[label] = rows
    return rows_by_sample


def collect_unique_contexts(
    rows_by_sample: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for sample_label, rows in rows_by_sample.items():
        for row in rows:
            context = normalize_context(row.get("doc", {}).get("context"))
            if not context:
                continue

            if context not in unique:
                unique[context] = {
                    "context_id": len(unique),
                    "context": context,
                    "doc_ids": [],
                    "sample_labels": [],
                    "TO_PAT_YN": row.get("doc", {}).get("TO_PAT_YN"),
                }

            unique[context]["doc_ids"].append(row.get("doc_id"))
            if sample_label not in unique[context]["sample_labels"]:
                unique[context]["sample_labels"].append(sample_label)

    return list(unique.values())


def build_system_prompt(use_prior: bool) -> str:
    prior_text = ""
    if use_prior:
        prior_text = (
            "\nThe expected aggregate source mix is roughly "
            "YNHH:Woven:Bethesda:Survey = 21:6:17:14. Treat this as a weak "
            "dataset-level prior only; do not force an individual label if the "
            "message clearly fits another source."
        )

    return (
        "You are a careful dataset-source judge for medical message contexts. "
        "Classify the target context into exactly one label: Bethesda, Survey, "
        "Woven, or YNHH.\n\n"
        "Bethesda, Survey, and Woven are represented by labeled examples. "
        "YNHH means the target context does not fit the Bethesda, Survey, or "
        "Woven source distributions and is best treated as the remaining YNHH "
        "dataset.\n"
        f"{prior_text}\n\n"
        "Use only the target context and the source examples. Do not infer from "
        "the model prediction or gold labels. Return one JSON object with keys: "
        "label, confidence, rationale. confidence must be one of high, medium, "
        "or low. rationale must be a short sentence."
    )


def build_user_prompt(
    source_examples: str,
    context_record: dict[str, Any],
    target_max_chars: int,
) -> str:
    target = truncate_text(context_record["context"], target_max_chars)
    return (
        f"{source_examples}\n\n"
        "Target context:\n"
        f"{target}\n\n"
        "Return JSON only, for example: "
        '{"label":"YNHH","confidence":"medium","rationale":"..."}'
    )


def parse_judgment(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        label_match = re.search(r"\b(Bethesda|Survey|Woven|YNHH)\b", content)
        if not label_match:
            raise
        parsed = {
            "label": label_match.group(1),
            "confidence": "low",
            "rationale": "Parsed label from non-JSON response.",
        }

    label = parsed.get("label")
    if label not in LABELS:
        raise ValueError(f"Invalid judge label: {label!r}")

    confidence = parsed.get("confidence", "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "label": label,
        "confidence": confidence,
        "rationale": str(parsed.get("rationale", "")).strip(),
    }


def make_client(api_key_env: str, api_key: str | None, base_url: str | None) -> OpenAI:
    resolved_api_key = api_key or os.environ.get(api_key_env)
    if not resolved_api_key:
        raise RuntimeError(
            f"No API key found. Set {api_key_env}, pass --api-key, or use a "
            "local OpenAI-compatible endpoint with --api-key dummy."
        )

    kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_judge(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return parse_judgment(content), content
        except Exception as exc:  # noqa: BLE001 - retry and surface final failure.
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep * (attempt + 1))

    raise RuntimeError(f"LLM judge failed after retries: {last_error}") from last_error


def load_existing_judgments(judgment_path: Path) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    if not judgment_path.exists():
        return judgments

    with judgment_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            context = normalize_context(row.get("context"))
            if context and row.get("label") in LABELS:
                judgments[context] = row
    return judgments


def write_judgment(judgment_path: Path, judgment: dict[str, Any]) -> None:
    judgment_path.parent.mkdir(parents=True, exist_ok=True)
    with judgment_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgment, ensure_ascii=True) + "\n")


def run_judging(
    contexts: list[dict[str, Any]],
    source_examples: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    judgments = load_existing_judgments(args.judgments)
    remaining = [item for item in contexts if item["context"] not in judgments]

    if args.limit is not None:
        remaining = remaining[: args.limit]

    system_prompt = build_system_prompt(args.use_prior)

    if args.dry_run:
        if not remaining:
            print("No remaining contexts for dry run.")
            return judgments
        prompt = build_user_prompt(source_examples, remaining[0], args.target_max_chars)
        print("SYSTEM PROMPT:\n")
        print(system_prompt)
        print("\nUSER PROMPT:\n")
        print(prompt)
        return judgments

    client = make_client(args.api_key_env, args.api_key, args.base_url)

    for index, context_record in enumerate(remaining, start=1):
        user_prompt = build_user_prompt(
            source_examples,
            context_record,
            args.target_max_chars,
        )
        parsed, raw_response = call_judge(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=not args.no_json_mode,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        judgment = {
            **context_record,
            **parsed,
            "raw_response": raw_response,
            "judge_model": args.model,
        }
        judgments[context_record["context"]] = judgment
        write_judgment(args.judgments, judgment)
        print(
            f"[{index}/{len(remaining)}] context_id={context_record['context_id']} "
            f"label={judgment['label']} confidence={judgment['confidence']}"
        )

    return judgments


def split_samples(
    rows_by_sample: dict[str, list[dict[str, Any]]],
    judgments: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    summary_rows = []
    for sample_label, rows in rows_by_sample.items():
        source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            context = normalize_context(row.get("doc", {}).get("context"))
            judgment = judgments.get(context)
            label = judgment["label"] if judgment else "UNJUDGED"

            split_row = dict(row)
            split_row["source_dataset"] = label
            split_row["source_judge"] = judgment
            split_row["source_match"] = "llm_judge_doc.context"
            source_rows[label].append(split_row)

        sample_output_dir = output_dir / sample_label
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        for label in list(LABELS) + ["UNJUDGED"]:
            label_rows = source_rows.get(label, [])
            output_path = sample_output_dir / f"{label}.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for row in label_rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            summary_rows.append(
                {
                    "sample": sample_label,
                    "source_dataset": label,
                    "n_rows": len(label_rows),
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
            "Use an LLM judge to split 70B lm-eval sample rows into Bethesda, "
            "Survey, Woven, and YNHH based on doc.context."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument(
        "samples",
        nargs="*",
        help="Sample JSONL paths, optionally as LABEL=PATH. Defaults to 70B raw and SFT samples.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key")
    parser.add_argument("--examples-per-source", type=int, default=8)
    parser.add_argument("--example-max-chars", type=int, default=450)
    parser.add_argument("--target-max-chars", type=int, default=1600)
    parser.add_argument("--limit", type=int, help="Judge only the first N unjudged contexts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt without calling an API.")
    parser.add_argument("--no-json-mode", action="store_true", help="Disable response_format=json_object.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--no-prior",
        dest="use_prior",
        action="store_false",
        help="Do not include the expected aggregate source-ratio prior in the judge prompt.",
    )
    parser.set_defaults(use_prior=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = dict(DEFAULT_SAMPLES)
    if args.samples:
        samples = dict(parse_sample_arg(sample) for sample in args.samples)

    for sample_path in samples.values():
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)

    contexts_by_source = load_source_contexts(args.source_dir)
    source_examples = build_source_examples(
        contexts_by_source,
        args.examples_per_source,
        args.example_max_chars,
    )
    rows_by_sample = load_sample_rows(samples)
    contexts = collect_unique_contexts(rows_by_sample)

    judgments = run_judging(contexts, source_examples, args)
    if args.dry_run:
        return

    summary_rows = split_samples(rows_by_sample, judgments, args.output_dir)
    summary_path = write_summary(summary_rows, args.output_dir)
    print_summary(summary_rows)
    print(f"\nWrote judgments: {args.judgments}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
