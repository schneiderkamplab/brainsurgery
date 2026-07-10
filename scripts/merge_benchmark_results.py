#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDS = [
    "axon",
    "checkpoint",
    "model_dir",
    "fallback",
    "masked_top1_eq",
    "masked_max_abs_diff",
    "masked_max_rel_diff",
    "hf_time",
    "axon_time",
    "speed_ratio_axon_over_hf",
    "max_len",
    "hf_time_norm128",
    "axon_time_norm128",
    "speed_ratio_axon_over_hf_norm128",
    "model_task",
    "param_count",
    "source_log",
    "result_json_path",
]

STALE_GPT2_PATH_RENAMES = {
    "/gpt2/generic-gpt2-kv.axon": "/gpt2/generic-gpt2.axon",
    "/gpt2/gpt2-kv.axon": "/gpt2/gpt2.axon",
    "/gpt2/gpt2-medium-kv.axon": "/gpt2/gpt2-medium.axon",
    "/gpt2/gpt2-large-kv.axon": "/gpt2/gpt2-large.axon",
    "/gpt2/gpt2-xl-kv.axon": "/gpt2/gpt2-xl.axon",
}


def _value(data: dict[str, Any], *names: str) -> str:
    for name in names:
        value = data.get(name)
        if value is not None:
            return str(value)
    return ""


def _load_result(path: Path, *, source_log: Path) -> dict[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    axon = data.get("axon_file") or data.get("axon") or data.get("axon_path")
    checkpoint = data.get("checkpoint_id") or data.get("checkpoint")
    if not axon or not checkpoint:
        return None
    original_axon = str(axon)
    return {
        "axon": str(axon),
        "_original_axon": original_axon,
        "checkpoint": str(checkpoint),
        "model_dir": _value(data, "hf_model_dir", "model_dir"),
        "fallback": _value(data, "fallback"),
        "masked_top1_eq": _value(data, "masked_top1_eq"),
        "masked_max_abs_diff": _value(data, "masked_max_diff", "masked_max_abs_diff"),
        "masked_max_rel_diff": _value(data, "masked_max_rel_diff"),
        "hf_time": _value(data, "hf_time", "hf_time_s"),
        "axon_time": _value(data, "axon_time", "axon_time_s"),
        "speed_ratio_axon_over_hf": _value(data, "speed_ratio_axon_over_hf"),
        "max_len": _infer_max_len(source_log),
        "model_task": _value(data, "model_task"),
        "param_count": _value(data, "param_count", "parameter_count", "num_parameters"),
        "source_log": str(source_log),
        "result_json_path": str(path),
    }


def _normalize_axon_path(axon: str) -> str:
    for old, new in STALE_GPT2_PATH_RENAMES.items():
        if axon.endswith(old):
            return f"{axon[: -len(old)]}{new}"
    return axon


def _drop_row(row: dict[str, str]) -> bool:
    checkpoint = row.get("checkpoint", "")
    axon = row.get("axon", "")
    if checkpoint.startswith("test/"):
        return True
    if axon.endswith("/gpt2/generic-gpt2-basic.axon"):
        return True
    return False


def _search(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return ""
    value = match.group(1)
    return value.strip() if value is not None else ""


def _load_text_result(path: Path, *, source_log: Path) -> dict[str, str] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    axon = _search(r"^Axon file:\s+(.+)$", text)
    checkpoint = _search(r"^Checkpoint:\s+(.+)$", text)
    if not axon or not checkpoint:
        return None
    hf_time = _search(r"^HF:\s+([0-9.eE+-]+)s total", text)
    axon_time = _search(r"^Axon-derived:\s+([0-9.eE+-]+)s total", text)
    if not hf_time:
        hf_time = _search(r"^HF \.\.\.\s+([0-9.eE+-]+)s$", text)
    if not axon_time:
        axon_time = _search(r"^AxonDerived \.\.\.\s+([0-9.eE+-]+)s$", text)
    ratio = ""
    if hf_time and axon_time:
        try:
            ratio = str(float(axon_time) / float(hf_time))
        except ValueError:
            ratio = ""
    return {
        "axon": axon,
        "_original_axon": axon,
        "checkpoint": checkpoint,
        "model_dir": _search(r"^Model dir:\s+(.+)$", text),
        "fallback": _search(r"^Fallback:\s+(.+)$", text),
        "masked_top1_eq": _search(r"^Masked top1_eq:\s+(.+)$", text),
        "masked_max_abs_diff": _search(r"^Masked abs diff \\(max\\):\s+(.+)$", text),
        "masked_max_rel_diff": _search(r"^Logits rel diff \\(masked\\) \\| mean/max:\s+[0-9.eE+-]+\s+([0-9.eE+-]+)$", text),
        "hf_time": hf_time,
        "axon_time": axon_time,
        "speed_ratio_axon_over_hf": ratio,
        "max_len": _infer_max_len(source_log),
        "model_task": _search(r"^Model task:\s+(.+)$", text),
        "param_count": "",
        "source_log": str(source_log),
        "result_json_path": str(path),
    }


def _load_rows(log_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for log_dir in log_dirs:
        json_keys: set[tuple[str, str]] = set()
        for path in sorted(log_dir.glob("**/*.result.json")):
            row = _load_result(path, source_log=log_dir)
            if row is not None:
                row["axon"] = _normalize_axon_path(row["axon"])
                if _drop_row(row):
                    continue
                json_keys.add((row["axon"], row["checkpoint"]))
                rows.append(row)
        for path in sorted(log_dir.glob("**/*.txt")):
            if path.name.startswith("parent-"):
                continue
            row = _load_text_result(path, source_log=log_dir)
            if row is not None:
                row["axon"] = _normalize_axon_path(row["axon"])
            if row is not None and not _drop_row(row) and (row["axon"], row["checkpoint"]) not in json_keys:
                rows.append(row)
    return rows


def _infer_max_len(source_log: Path) -> str:
    name = source_log.name.lower()
    if "1024" in name:
        return "1024"
    if "maxlen128" in name or "max-len-128" in name or "128" in name:
        return "128"
    # Current operational convention: targeted fixes without an explicit
    # maxlen1024 marker are smoke/perf rows run with max-len 128.
    return "128"


def _add_normalized_128(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    if not row.get("model_task"):
        row["model_task"] = _infer_model_task(row)
    max_len_raw = row.get("max_len", "")
    try:
        max_len = float(max_len_raw)
    except ValueError:
        max_len = 128.0
    factor = max_len / 128.0 if max_len > 0 else 1.0
    for src, dst in (("hf_time", "hf_time_norm128"), ("axon_time", "axon_time_norm128")):
        try:
            row[dst] = str(float(row.get(src, "")) / factor)
        except ValueError:
            row[dst] = ""
    try:
        hf = float(row["hf_time_norm128"])
        axon = float(row["axon_time_norm128"])
        row["speed_ratio_axon_over_hf_norm128"] = str(axon / hf) if hf > 0 else ""
    except ValueError:
        row["speed_ratio_axon_over_hf_norm128"] = ""
    return row


def _infer_model_task(row: dict[str, str]) -> str:
    axon = row.get("axon", "").lower()
    checkpoint = row.get("checkpoint", "").lower()
    try:
        text = Path(row.get("axon", "")).read_text(encoding="utf-8")
    except OSError:
        text = ""
    match = re.search(r'\{-#\s*TASK\s+"([^"]+)"\s*#-\}', text)
    if match:
        return match.group(1)
    if "/t5gemma" not in axon and "t5gemma" not in checkpoint:
        if "/gemma" in axon or "google/gemma" in checkpoint:
            return "causal_lm"
    return ""


def _merge_latest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    for row in rows:
        key = (row["axon"], row["checkpoint"])
        priority = _row_priority(row)
        previous = latest.get(key)
        if previous is None or priority >= previous[0]:
            latest[key] = (priority, row)
    return [_add_normalized_128(row) for _, row in latest.values()]



def _row_priority(row: dict[str, str]) -> int:
    original = row.get("_original_axon", row.get("axon", ""))
    if original.endswith("/gpt2/generic-gpt2-kv.axon"):
        return 20
    if original.endswith("/gpt2/gpt2-kv.axon"):
        return 20
    if original.endswith("/gpt2/gpt2-medium-kv.axon"):
        return 20
    if original.endswith("/gpt2/gpt2-large-kv.axon"):
        return 20
    if original.endswith("/gpt2/gpt2-xl-kv.axon"):
        return 20
    if original.endswith("/gpt2/generic-gpt2.axon"):
        return 5
    return 10


def _relative_axon(row: dict[str, str]) -> str:
    marker = "/brainsurgery/synapse/models/"
    axon = row["axon"]
    if marker in axon:
        return axon.split(marker, 1)[1]
    return axon


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge axon-benchmark result JSON logs into one latest-row CSV.")
    parser.add_argument("log_dirs", nargs="+", type=Path, help="Log directories in precedence order; later dirs override earlier dirs for the same Axon/checkpoint.")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    rows = _merge_latest(_load_rows(args.log_dirs))
    rows.sort(key=lambda row: (_relative_axon(row), row["checkpoint"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})

    timed = sum(1 for row in rows if row.get("hf_time") and row.get("axon_time"))
    errors = sum(1 for row in rows if row.get("masked_top1_eq") == "ERROR")
    print(f"wrote {args.output} ({len(rows)} rows, {timed} timed, {errors} errors)")


if __name__ == "__main__":
    main()
