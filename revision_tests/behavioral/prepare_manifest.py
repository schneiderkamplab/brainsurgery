#!/usr/bin/env python3
"""Build the frozen behavioral prompt manifest from pinned upstream files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
SOURCES_PATH = HERE / "sources.yaml"
OUTPUT_PATH = HERE / "prompt_manifest.jsonl"
SUMMARY_PATH = HERE / "manifest_summary.json"
PROTOCOL_ID = "eacl2027_behavioral_v1"
LABELS = ("A", "B", "C", "D")
BELEBELE_INSTRUCTION = (
    "Given the following passage, query, and answer choices, output the letter "
    "corresponding to the correct answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belebele-archive", type=Path, required=True)
    parser.add_argument("--belebele-dir", type=Path, required=True)
    parser.add_argument("--mmlu-parquet", type=Path, required=True)
    parser.add_argument("--human-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(SOURCES_PATH.read_text(encoding="utf-8"))
    require_config(config)
    sources = config["sources"]
    verify_sha(args.belebele_archive, sources["belebele"]["archive_sha256"])
    verify_sha(args.mmlu_parquet, sources["mmlu"]["data_sha256"])
    verify_sha(args.human_eval, sources["human_eval"]["data_sha256"])

    seed = config["selection_seed"]
    rows = [
        *build_belebele_rows(args.belebele_dir, config, seed),
        *build_mmlu_rows(args.mmlu_parquet, config, seed),
        *build_human_eval_rows(args.human_eval, config, seed),
    ]
    for ordinal, row in enumerate(rows, start=1):
        row["ordinal"] = ordinal

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    args.output.write_text(serialized, encoding="utf-8")
    summary = build_summary(rows, config, args.output)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} prompts to {args.output}")
    print(f"manifest sha256: {summary['manifest_sha256']}")
    return 0


def require_config(config: Any) -> None:
    if not isinstance(config, dict) or config.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit(f"sources.yaml must declare {PROTOCOL_ID}")


def build_belebele_rows(root: Path, config: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    source = config["sources"]["belebele"]
    languages = config["strata"]["belebele_languages"]
    count = config["sample_sizes"]["belebele_parallel_question_ids"]
    raw_by_language: dict[str, list[dict[str, Any]]] = {}
    indexed_by_language: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for dialect in languages:
        path = root / f"{dialect}.jsonl"
        raw_rows = read_jsonl(path)
        raw_by_language[dialect] = raw_rows
        indexed_by_language[dialect] = {
            belebele_item_id(row): (index, row) for index, row in enumerate(raw_rows)
        }

    english = raw_by_language["eng_Latn"]
    candidates = [(belebele_item_id(row), row) for row in english]
    selected = select_ranked(candidates, seed, "belebele", "parallel_question", count)
    selected_ids = [item_id for _, item_id, _ in selected]
    for dialect, index in indexed_by_language.items():
        missing = [item_id for item_id in selected_ids if item_id not in index]
        if missing:
            raise SystemExit(f"Belebele {dialect} lacks selected parallel ids: {missing}")

    result: list[dict[str, Any]] = []
    for dialect, language_info in languages.items():
        for rank, (rank_hash, item_id, _) in enumerate(selected, start=1):
            row_index, row = indexed_by_language[dialect][item_id]
            prompt = normalize_text(render_belebele(row))
            answer_index = int(row["correct_answer_num"]) - 1
            result.append(
                manifest_row(
                    prompt_id=f"belebele_{dialect}_{rank:02d}",
                    source_key="belebele",
                    source=source,
                    source_item_id=item_id,
                    source_row_index=row_index,
                    selection_stratum=f"parallel_question:{dialect}",
                    selection_rank=rank,
                    selection_rank_sha256=rank_hash,
                    language_code=dialect,
                    language=language_info["language"],
                    script=language_info["script"],
                    task_category="multilingual_reading_comprehension",
                    prompt_template_id="belebele_zero_shot_en_instruction_v1",
                    prompt=prompt,
                    expected={
                        "kind": "multiple_choice",
                        "labels": list(LABELS),
                        "correct_label": LABELS[answer_index],
                    },
                )
            )
    return result


def build_mmlu_rows(path: Path, config: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MMLU manifest generation requires pyarrow; use an isolated environment, e.g. "
            "uv run --isolated --with pyarrow --with pyyaml python ..."
        ) from exc

    source = config["sources"]["mmlu"]
    subjects = config["strata"]["mmlu_subjects"]
    count = config["sample_sizes"]["mmlu_per_subject"]
    table_rows = parquet.read_table(path).to_pylist()
    result: list[dict[str, Any]] = []
    for subject, broad_category in subjects.items():
        candidates = []
        subject_index = 0
        for global_index, row in enumerate(table_rows):
            if row["subject"] != subject:
                continue
            item_id = f"{subject}:test:{subject_index}"
            candidates.append((item_id, (global_index, subject_index, row)))
            subject_index += 1
        selected = select_ranked(candidates, seed, "mmlu", subject, count)
        for rank, (rank_hash, item_id, payload) in enumerate(selected, start=1):
            global_index, subject_row_index, row = payload
            choices = [normalize_text(str(choice)) for choice in row["choices"]]
            answer_index = int(row["answer"])
            prompt = normalize_text(render_mmlu(row["question"], choices))
            result.append(
                manifest_row(
                    prompt_id=f"mmlu_{subject}_{rank:02d}",
                    source_key="mmlu",
                    source=source,
                    source_item_id=item_id,
                    source_row_index=global_index,
                    selection_stratum=subject,
                    selection_rank=rank,
                    selection_rank_sha256=rank_hash,
                    language_code="eng_Latn",
                    language="English",
                    script="Latin",
                    task_category=f"knowledge_reasoning:{broad_category}",
                    prompt_template_id="mmlu_zero_shot_v1",
                    prompt=prompt,
                    expected={
                        "kind": "multiple_choice",
                        "labels": list(LABELS),
                        "correct_label": LABELS[answer_index],
                    },
                    extra={"subject_row_index": subject_row_index},
                )
            )
    return result


def build_human_eval_rows(path: Path, config: dict[str, Any], seed: str) -> list[dict[str, Any]]:
    source = config["sources"]["human_eval"]
    count = config["sample_sizes"]["human_eval"]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]
    candidates = [(row["task_id"], (index, row)) for index, row in enumerate(raw_rows)]
    selected = select_ranked(candidates, seed, "human_eval", "python", count)
    result = []
    for rank, (rank_hash, item_id, payload) in enumerate(selected, start=1):
        row_index, row = payload
        prompt = normalize_text(row["prompt"])
        result.append(
            manifest_row(
                prompt_id=f"human_eval_{rank:02d}",
                source_key="human_eval",
                source=source,
                source_item_id=item_id,
                source_row_index=row_index,
                selection_stratum="python",
                selection_rank=rank,
                selection_rank_sha256=rank_hash,
                language_code="eng_Latn",
                language="English",
                script="Latin",
                task_category="python_code_completion",
                prompt_template_id="human_eval_upstream_prompt_v1",
                prompt=prompt,
                expected={"kind": "regression_only"},
                extra={"entry_point": row["entry_point"]},
            )
        )
    return result


def manifest_row(
    *,
    prompt_id: str,
    source_key: str,
    source: dict[str, Any],
    source_item_id: str,
    source_row_index: int,
    selection_stratum: str,
    selection_rank: int,
    selection_rank_sha256: str,
    language_code: str,
    language: str,
    script: str,
    task_category: str,
    prompt_template_id: str,
    prompt: str,
    expected: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "protocol_id": PROTOCOL_ID,
        "prompt_id": prompt_id,
        "source": source_key,
        "source_name": source["name"],
        "source_item_id": source_item_id,
        "source_row_index_zero_based": source_row_index,
        "source_revision": source.get("data_revision", source["repository_revision"]),
        "split": source["split"],
        "license": source["license"],
        "citation_key": source["citation_key"],
        "selection_stratum": selection_stratum,
        "selection_rank_one_based": selection_rank,
        "selection_rank_sha256": selection_rank_sha256,
        "language_code": language_code,
        "language": language,
        "script": script,
        "instruction_language": "English" if source_key == "belebele" else language,
        "task_category": task_category,
        "prompt_template_id": prompt_template_id,
        "normalization": "Unicode_NFC_and_LF",
        "prompt": prompt,
        "prompt_sha256": text_sha256(prompt),
        "expected": expected,
    }
    if extra:
        row.update(extra)
    return row


def select_ranked(
    candidates: list[tuple[str, Any]], seed: str, source: str, stratum: str, count: int
) -> list[tuple[str, str, Any]]:
    ranked = [
        (text_sha256("\0".join((seed, source, stratum, item_id))), item_id, payload)
        for item_id, payload in candidates
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) < count:
        raise SystemExit(f"{source}/{stratum} has {len(ranked)} candidates, needs {count}")
    return ranked[:count]


def render_belebele(row: dict[str, Any]) -> str:
    choices = "\n".join(
        f"({label}) {row[f'mc_answer{index}']}" for index, label in enumerate(LABELS, start=1)
    )
    return (
        f"{BELEBELE_INSTRUCTION}\n###\nPassage:\n{row['flores_passage']}\n###\n"
        f"Query:\n{row['question']}\n###\nChoices:\n{choices}\n###\nAnswer:\n"
    )


def render_mmlu(question: str, choices: list[str]) -> str:
    rendered_choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(LABELS, choices, strict=True)
    )
    return f"Question: {question}\n{rendered_choices}\nAnswer:"


def belebele_item_id(row: dict[str, Any]) -> str:
    return f"{row['link']}|question_{row['question_number']}"


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_sha(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise SystemExit(f"source checksum mismatch for {path}: expected {expected}, got {actual}")


def build_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "manifest": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
        "prompt_count": len(rows),
        "multiple_choice_count": sum(row["expected"]["kind"] == "multiple_choice" for row in rows),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "language_counts": dict(sorted(Counter(row["language_code"] for row in rows).items())),
        "task_category_counts": dict(sorted(Counter(row["task_category"] for row in rows).items())),
        "license_counts": dict(sorted(Counter(row["license"] for row in rows).items())),
        "selection_seed": config["selection_seed"],
        "source_fingerprints": config["sources"],
    }


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
