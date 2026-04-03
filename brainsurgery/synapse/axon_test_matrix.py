from __future__ import annotations

import argparse
import contextlib
import gc
import io
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .axon_test import run_axon_test


@dataclass(frozen=True)
class _Pair:
    axon_path: Path
    model_dir: Path


@dataclass(frozen=True)
class _SummaryRow:
    axon_file: str
    model_dir: str
    hf_runtime_s: str
    axon_runtime_s: str
    runtime_ratio: str
    eval_max_abs_diff: str
    eval_max_rel_diff: str
    debug_max_logit_diff: str
    debug_max_rel_diff: str
    mean_rel_diff: str
    masked_max_diff: str
    masked_last_max_diff: str
    masked_mean_rel_diff: str
    masked_max_rel_diff: str
    eval_top1_eq: str
    debug_top1_eq: str
    masked_top1_eq: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run brainsurgery synapse axon-test across matching examples/*.axon and models/* dirs."
        )
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("examples"),
        help="Directory with Axon files (default: examples).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory with model directories (default: models).",
    )
    parser.add_argument("--device", default="cpu", help="Device passed to axon-test.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Floating point dtype passed to axon-test.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=32,
        help="Total sequence length target for generation.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Prompt text. Repeat to pass multiple prompts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-run output from synapse axon-test.",
    )
    parser.add_argument(
        "--no-capture-output",
        action="store_true",
        help="Do not capture per-run output; stream run_axon_test output directly.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Only run pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "Exclude pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve and print matching pairs; do not run tests.",
    )
    parser.add_argument(
        "--table-format",
        default="plain",
        choices=["plain", "markdown"],
        help="Summary table format (plain or markdown).",
    )
    parser.add_argument(
        "--compile-hf",
        action="store_true",
        help="Compile the HF reference model with torch.compile.",
    )
    parser.add_argument(
        "--compile-axon",
        action="store_true",
        help="Compile the Axon-derived model with torch.compile.",
    )
    parser.add_argument(
        "--compile-backend",
        default=None,
        help="Optional torch.compile backend (e.g. inductor).",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional torch.compile mode (e.g. default/reduce-overhead/max-autotune).",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Set torch.compile(fullgraph=True).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Set torch.compile(dynamic=True).",
    )
    parser.add_argument(
        "--model-task",
        default="auto",
        choices=["auto", "causal_lm", "masked_lm", "seq2seq_lm"],
        help=(
            "Execution task for run_axon_test. "
            "'auto' picks per model (encoder-only masked LM models -> masked_lm, "
            "t5_small=seq2seq_lm, others=causal_lm)."
        ),
    )
    return parser.parse_args()


def _normalize_selectors(values: list[str] | None) -> set[str]:
    out: set[str] = set()
    if not values:
        return out
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        out.add(Path(raw).name)
    return out


def _pair_matches_selector(pair: _Pair, selectors: set[str]) -> bool:
    if not selectors:
        return False
    axon_name = pair.axon_path.name.lower()
    model_name = pair.model_dir.name.lower()
    for selector in selectors:
        if selector.endswith(".axon"):
            if selector == axon_name:
                return True
            continue
        if selector == model_name:
            return True
    return False


def _apply_pair_filters(
    pairs: list[_Pair],
    *,
    include: set[str],
    exclude: set[str],
) -> list[_Pair]:
    if include and exclude:
        raise ValueError("axon-test-matrix accepts either include or exclude selectors, not both")
    if include:
        return [pair for pair in pairs if _pair_matches_selector(pair, include)]
    if exclude:
        return [pair for pair in pairs if not _pair_matches_selector(pair, exclude)]
    return pairs


def _resolve_model_task_for_pair(pair: _Pair) -> str:
    masked_lm_stems = {
        "albert",
        "bert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
    }
    masked_lm_model_dirs = {
        "albert",
        "bert",
        "camembert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
        "xlm_roberta",
    }
    seq2seq_lm_stems = {
        "t5",
        "t5_small",
        "mt5",
        "bart",
        "mbart",
        "marian",
        "t5gemma",
        "t5gemma2",
    }
    seq2seq_lm_model_dirs = {
        "t5_small",
        "t5_base",
        "t5_large",
        "t5_3b",
        "t5_11b",
        "mt5_small",
        "bart_base",
        "mbart_large_50_m2m",
        "marian_en_de",
        "t5gemma_s_s_ul2",
        "t5gemma2_270m",
    }
    if pair.axon_path.stem in masked_lm_stems or pair.model_dir.name in masked_lm_model_dirs:
        return "masked_lm"
    if (
        pair.axon_path.stem in seq2seq_lm_stems
        or pair.model_dir.name in seq2seq_lm_model_dirs
        or pair.model_dir.name.startswith("t5gemma")
    ):
        return "seq2seq_lm"
    return "causal_lm"


def _resolve_pairs(
    examples_dir: Path,
    models_dir: Path,
) -> list[_Pair]:
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    excluded_model_dir_names = {
        "flexolmo",
        "glm_4_5_air",
        "jamba_tiny_random",
        "nemotron3",
        "nemotron-3",
        "test",
    }
    model_dirs = sorted(
        path
        for path in models_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in excluded_model_dir_names
    )
    model_by_name = {path.name: path for path in model_dirs}
    explicit_model_aliases = {
        "flexolmo": "flexmath",
        "black_mamba": "black_mamba_2_8b",
        "mamba": "mamba_tiny_random",
        "mamba_2_8b": "mamba_2_8b_hf",
        "jamba": "jamba_3b",
        "gemma_1b": "gemma3_1b",
        "gemma3_270m": "gemma3",
        "gemma3_config": "gemma3",
        "olmo3": "olmo3_7b_instruct",
        "phi3": "phi3_mini_4k_instruct",
        "smollm": "smollm_135m",
        "smollm3": "smollm3_3b_base",
        "smollm3_config": "smollm3_3b_base",
        "t5": "t5_base",
        "phi3minimedium": "phi3_mini_4k_instruct",
        "phi3small": "phi3_small_8k_instruct",
        "olmo2": "olmo_2_1b",
        "olmo2_config": "olmo_2_1b",
        "mt5": "mt5_small",
        "bart": "bart_base",
        "mbart": "mbart_large_50_m2m",
        "marian": "marian_en_de",
        "t5gemma": "t5gemma_s_s_ul2",
        "t5gemma2": "t5gemma2_270m",
    }
    excluded_stems = {
        "glm_4_5_air",
        "nemotron-3",
        "nemotron3",
    }

    def _resolve_model_dir_for_axon_stem(stem: str) -> Path | None:
        model_dir = model_by_name.get(explicit_model_aliases.get(stem, stem))
        if model_dir is not None:
            return model_dir
        parts = stem.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = "_".join(parts[:cut])
            model_dir = model_by_name.get(candidate)
            if model_dir is not None:
                return model_dir
        return None

    axon_paths = sorted(
        path for path in examples_dir.glob("*.axon") if not path.stem.endswith("_config")
    )
    axon_by_stem = {path.stem: path for path in axon_paths}

    pairs: list[_Pair] = []
    covered_model_dirs: set[Path] = set()

    # Pass 1: include every resolvable Axon example exactly once.
    for axon_path in axon_paths:
        stem = axon_path.stem
        if stem in excluded_stems:
            continue
        model_dir = _resolve_model_dir_for_axon_stem(stem)

        if model_dir is not None:
            pairs.append(_Pair(axon_path=axon_path, model_dir=model_dir))
            covered_model_dirs.add(model_dir)
        else:
            print(f"Ignoring {axon_path} as I did not locate model_dir from stem {stem}")

    # Pass 2: ensure every model dir has at least one Axon file assigned.
    explicit_axon_aliases = {
        "flexmath": "flexolmo",
        "black_mamba_2_8b": "black_mamba",
        "camembert": "roberta",
        "comma": "dfm_decoder",
        "mamba_tiny_random": "mamba",
        "mamba_2_8b_hf": "mamba_2_8b",
        "jamba_3b": "jamba_3b",
        "gpt2": "gpt2",
        "gemma3": "gemma3",
        "gemma3_1b": "gemma_1b",
        "gemma3_4b": "gemma3",
        "gemma3_12b": "gemma3",
        "smollm_135m": "smollm",
        "smollm_360m": "smollm",
        "smollm_1_7b": "smollm",
        "smollm2_135m": "smollm",
        "smollm2_360m": "smollm",
        "smollm2_1_7b": "smollm",
        "smollm3_3b": "smollm3",
        "smollm3_3b_base": "smollm3",
        "phi3_mini_4k_instruct": "phi3minimedium",
        "phi3_mini_128k_instruct": "phi3minimedium",
        "phi3_medium_4k_instruct": "phi3minimedium",
        "phi3_medium_128k_instruct": "phi3minimedium",
        "phi3_small_8k_instruct": "phi3small",
        "phi3_small_128k_instruct": "phi3small",
        "olmo_2_1b": "olmo2",
        "olmo_2_7b": "olmo2",
        "olmo_2_13b": "olmo2",
        "xlm_roberta": "roberta",
        "mt5_small": "mt5",
        "bart_base": "bart",
        "mbart_large_50_m2m": "mbart",
        "marian_en_de": "marian",
        "t5gemma_s_s_ul2": "t5gemma",
        "t5gemma2_270m": "t5gemma2",
    }

    for model_dir in model_dirs:
        if model_dir in covered_model_dirs:
            continue
        model_name = model_dir.name
        candidate_stems: list[str] = []
        alias = explicit_axon_aliases.get(model_name)
        if alias is not None:
            candidate_stems.append(alias)
        candidate_stems.append(model_name)
        parts = model_name.split("_")
        for cut in range(len(parts) - 1, 0, -1):
            candidate_stems.append("_".join(parts[:cut]))
        resolved_axon: Path | None = None
        for candidate in candidate_stems:
            resolved_axon = axon_by_stem.get(candidate)
            if resolved_axon is not None:
                break
        if resolved_axon is None:
            print(f"Ignoring model dir {model_dir} as I did not locate matching .axon file")
            continue
        if resolved_axon.stem in excluded_stems:
            continue
        # Avoid accidental duplicate model+axon pair.
        duplicate = any(
            pair.axon_path == resolved_axon and pair.model_dir == model_dir for pair in pairs
        )
        if duplicate:
            continue
        pairs.append(_Pair(axon_path=resolved_axon, model_dir=model_dir))
        covered_model_dirs.add(model_dir)

    return pairs


def _format_table(rows: list[_SummaryRow]) -> str:
    headers = [
        "axon_file",
        "model_dir",
        "HF runtime (s)",
        "AxonDerived runtime (s)",
        "AxonDerived runtime/HF runtime",
        "eval max abs diff",
        "eval max rel diff",
        "eval top1_eq",
        "masked max abs diff",
        "masked last max abs diff",
        "masked max rel diff",
        "masked_top1_eq",
        "debug max abs diff",
        "debug max rel diff",
        "debug top1_eq",
        "mean rel diff",
        "masked mean rel diff",
    ]

    body = [
        [
            row.axon_file,
            row.model_dir,
            row.hf_runtime_s,
            row.axon_runtime_s,
            row.runtime_ratio,
            row.eval_max_abs_diff,
            row.eval_max_rel_diff,
            row.eval_top1_eq,
            row.masked_max_diff,
            row.masked_last_max_diff,
            row.masked_max_rel_diff,
            row.masked_top1_eq,
            row.debug_max_logit_diff,
            row.debug_max_rel_diff,
            row.debug_top1_eq,
            row.mean_rel_diff,
            row.masked_mean_rel_diff,
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for line in body:
        for idx, cell in enumerate(line):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt(line: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(line))

    divider = "-+-".join("-" * width for width in widths)
    out_lines = [_fmt(headers), divider]
    out_lines.extend(_fmt(line) for line in body)
    return "\n".join(out_lines)


def _format_table_markdown(rows: list[_SummaryRow]) -> str:
    headers = [
        "axon_file",
        "model_dir",
        "HF runtime (s)",
        "AxonDerived runtime (s)",
        "AxonDerived runtime/HF runtime",
        "eval max abs diff",
        "eval max rel diff",
        "eval top1_eq",
        "masked max abs diff",
        "masked last max abs diff",
        "masked max rel diff",
        "masked_top1_eq",
        "debug max abs diff",
        "debug max rel diff",
        "debug top1_eq",
        "mean rel diff",
        "masked mean rel diff",
    ]

    body = [
        [
            row.axon_file,
            row.model_dir,
            row.hf_runtime_s,
            row.axon_runtime_s,
            row.runtime_ratio,
            row.eval_max_abs_diff,
            row.eval_max_rel_diff,
            row.eval_top1_eq,
            row.masked_max_diff,
            row.masked_last_max_diff,
            row.masked_max_rel_diff,
            row.masked_top1_eq,
            row.debug_max_logit_diff,
            row.debug_max_rel_diff,
            row.debug_top1_eq,
            row.mean_rel_diff,
            row.masked_mean_rel_diff,
        ]
        for row in rows
    ]

    def _esc(cell: str) -> str:
        return cell.replace("|", r"\|")

    header_row = "| " + " | ".join(_esc(h) for h in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = ["| " + " | ".join(_esc(cell) for cell in line) + " |" for line in body]
    return "\n".join([header_row, divider, *data_rows])


def _run_pair(
    pair: _Pair,
    *,
    model_task: str,
    device: str,
    dtype: str,
    max_len: int,
    text: list[str],
    verbose: bool,
    no_capture_output: bool,
    compile_hf: bool,
    compile_axon: bool,
    compile_backend: str | None,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "axon_file": pair.axon_path,
        "weights": pair.model_dir,
        "hf_model_dir": pair.model_dir,
        "device": device,
        "dtype": dtype,
        "model_task": model_task,
        "max_len": max_len,
        "text": text,
        "compile_hf": compile_hf,
        "compile_axon": compile_axon,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "compile_dynamic": compile_dynamic,
    }
    if verbose:
        print(f"Running: {kwargs}")
    if no_capture_output:
        return run_axon_test(**kwargs)

    with contextlib.redirect_stdout(io.StringIO()):
        return run_axon_test(**kwargs)


def run_axon_test_matrix(
    *,
    examples_dir: Path = Path("examples"),
    models_dir: Path = Path("models"),
    device: str = "cpu",
    dtype: str = "float32",
    max_len: int = 32,
    text: list[str] | None = None,
    verbose: bool = False,
    no_capture_output: bool = False,
    dry_run: bool = False,
    table_format: str = "plain",
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    model_task_override: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> int:
    if table_format not in {"plain", "markdown"}:
        raise ValueError("table_format must be 'plain' or 'markdown'")

    prompts = text if text else ["The future of AI is"]
    include_selectors = _normalize_selectors(include)
    exclude_selectors = _normalize_selectors(exclude)
    if model_task_override is not None and model_task_override not in {
        "causal_lm",
        "masked_lm",
        "seq2seq_lm",
    }:
        raise ValueError("model_task_override must be one of: causal_lm, masked_lm, seq2seq_lm")
    if include_selectors and exclude_selectors:
        raise ValueError("axon-test-matrix accepts either include or exclude selectors, not both")
    pairs = _resolve_pairs(examples_dir.resolve(), models_dir.resolve())
    pairs = _apply_pair_filters(pairs, include=include_selectors, exclude=exclude_selectors)
    if not pairs:
        print("No matching .axon/model directory pairs found.")
        return 1

    if dry_run:
        dry_rows = [
            _SummaryRow(
                axon_file=pair.axon_path.name,
                model_dir=str(pair.model_dir),
                hf_runtime_s="DRY-RUN",
                axon_runtime_s="DRY-RUN",
                runtime_ratio="DRY-RUN",
                eval_max_abs_diff="DRY-RUN",
                eval_max_rel_diff="DRY-RUN",
                debug_max_logit_diff="DRY-RUN",
                debug_max_rel_diff="DRY-RUN",
                mean_rel_diff="DRY-RUN",
                masked_max_diff="DRY-RUN",
                masked_last_max_diff="DRY-RUN",
                masked_mean_rel_diff="DRY-RUN",
                masked_max_rel_diff="DRY-RUN",
                eval_top1_eq="DRY-RUN",
                debug_top1_eq="DRY-RUN",
                masked_top1_eq="DRY-RUN",
            )
            for pair in pairs
        ]
        if table_format == "markdown":
            print(_format_table_markdown(dry_rows))
        else:
            print(_format_table(dry_rows))
        return 0

    rows: list[_SummaryRow] = []
    passed = 0
    failed = 0

    progress = tqdm(total=len(pairs), desc="synapse axon-test", unit="pair")
    for pair in pairs:
        progress.set_postfix_str(pair.axon_path.name)
        try:
            model_task = model_task_override or _resolve_model_task_for_pair(pair)
            result = _run_pair(
                pair,
                model_task=model_task,
                device=device,
                dtype=dtype,
                max_len=max_len,
                text=prompts,
                verbose=verbose,
                no_capture_output=no_capture_output,
                compile_hf=compile_hf,
                compile_axon=compile_axon,
                compile_backend=compile_backend,
                compile_mode=compile_mode,
                compile_fullgraph=compile_fullgraph,
                compile_dynamic=compile_dynamic,
            )
            masked_max_diff_value = result.get("masked_max_diff")
            masked_max_rel_diff_value = result.get("masked_max_rel_diff")
            masked_top1_eq_value = result.get("masked_top1_eq")
            eval_max_abs_diff_value = (
                masked_max_diff_value if masked_max_diff_value is not None else result["max_diff"]
            )
            eval_max_rel_diff_value = (
                masked_max_rel_diff_value
                if masked_max_rel_diff_value is not None
                else result["max_rel_diff"]
            )
            eval_top1_eq_value = (
                bool(masked_top1_eq_value)
                if masked_top1_eq_value is not None
                else bool(result["top1_eq"])
            )
            rows.append(
                _SummaryRow(
                    axon_file=pair.axon_path.name,
                    model_dir=str(pair.model_dir),
                    hf_runtime_s=f"{result['hf_time']:.6g}",
                    axon_runtime_s=f"{result['axon_time']:.6g}",
                    runtime_ratio=f"{result['speed_ratio_axon_over_hf']:.3f}",
                    eval_max_abs_diff=f"{float(eval_max_abs_diff_value):.6g}",
                    eval_max_rel_diff=f"{float(eval_max_rel_diff_value):.6g}",
                    debug_max_logit_diff=f"{result['max_diff']:.6g}",
                    debug_max_rel_diff=f"{float(result['max_rel_diff']):.6g}",
                    mean_rel_diff=f"{float(result['mean_rel_diff']):.6g}",
                    masked_max_diff=(
                        "N/A"
                        if result.get("masked_max_diff") is None
                        else f"{float(result['masked_max_diff']):.6g}"
                    ),
                    masked_last_max_diff=(
                        "N/A"
                        if result.get("masked_last_max_diff") is None
                        else f"{float(result['masked_last_max_diff']):.6g}"
                    ),
                    masked_mean_rel_diff=(
                        "N/A"
                        if result.get("masked_mean_rel_diff") is None
                        else f"{float(result['masked_mean_rel_diff']):.6g}"
                    ),
                    masked_max_rel_diff=(
                        "N/A"
                        if result.get("masked_max_rel_diff") is None
                        else f"{float(result['masked_max_rel_diff']):.6g}"
                    ),
                    eval_top1_eq=str(eval_top1_eq_value),
                    debug_top1_eq=str(bool(result["top1_eq"])),
                    masked_top1_eq=(
                        "N/A"
                        if result.get("masked_top1_eq") is None
                        else str(bool(result["masked_top1_eq"]))
                    ),
                )
            )
            passed += 1
        except Exception as exc:
            if verbose:
                tqdm.write(
                    f"ERROR in {pair.axon_path.name} ({pair.model_dir}): {type(exc).__name__}: {exc}"
                )
                for line in traceback.format_exc().rstrip().splitlines():
                    tqdm.write(line)
            rows.append(
                _SummaryRow(
                    axon_file=pair.axon_path.name,
                    model_dir=str(pair.model_dir),
                    hf_runtime_s="ERROR",
                    axon_runtime_s="ERROR",
                    runtime_ratio="ERROR",
                    eval_max_abs_diff="ERROR",
                    eval_max_rel_diff="ERROR",
                    debug_max_logit_diff="ERROR",
                    debug_max_rel_diff="ERROR",
                    mean_rel_diff="ERROR",
                    masked_max_diff="ERROR",
                    masked_last_max_diff="ERROR",
                    masked_mean_rel_diff="ERROR",
                    masked_max_rel_diff="ERROR",
                    eval_top1_eq=f"ERROR: {type(exc).__name__}: {exc}",
                    debug_top1_eq="ERROR",
                    masked_top1_eq="ERROR",
                )
            )
            failed += 1
        finally:
            gc.collect()
            progress.update(1)
            progress.set_postfix_str(f"passed={passed} failed={failed}")

    progress.close()

    print()
    if table_format == "markdown":
        print(_format_table_markdown(rows))
    else:
        print(_format_table(rows))
    print()
    print(f"Total: {len(pairs)} | Passed: {passed} | Failed: {failed}")
    return 0 if failed == 0 else 2


def main() -> int:
    args = _parse_args()
    model_task_override = None if args.model_task == "auto" else str(args.model_task)
    return run_axon_test_matrix(
        examples_dir=args.examples_dir,
        models_dir=args.models_dir,
        device=args.device,
        dtype=args.dtype,
        max_len=args.max_len,
        text=args.text,
        verbose=args.verbose,
        no_capture_output=bool(args.no_capture_output),
        dry_run=args.dry_run,
        table_format=args.table_format,
        compile_hf=bool(args.compile_hf),
        compile_axon=bool(args.compile_axon),
        compile_backend=(str(args.compile_backend) if args.compile_backend is not None else None),
        compile_mode=str(args.compile_mode) if args.compile_mode is not None else None,
        compile_fullgraph=bool(args.compile_fullgraph),
        compile_dynamic=bool(args.compile_dynamic),
        model_task_override=model_task_override,
        include=args.include,
        exclude=args.exclude,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_axon_test_matrix"]
