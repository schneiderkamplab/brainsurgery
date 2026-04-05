from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf
from typer.models import OptionInfo

app = typer.Typer(help="Synapse tooling.")


def _synapse_module() -> Any:
    return importlib.import_module("brainsurgery.synapse")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Expected YAML mapping at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in data.items()}


def _emit_model_code(spec: dict[str, Any], class_name: str) -> str:
    module = _synapse_module()
    emit_fn = getattr(module, "emit_model_code_from_synapse_spec")
    return emit_fn(spec, class_name=class_name)


def _parse_axon_to_synapse_spec(
    axon_path: Path, *, main_module: str | None = None
) -> dict[str, Any]:
    module = _synapse_module()
    parse_fn = getattr(module, "parse_axon_program_from_path")
    lower_fn = getattr(module, "lower_axon_program_to_synapse_spec")
    parsed = parse_fn(axon_path)
    return lower_fn(parsed, main_module=main_module)


def _render_synapse_to_axon_text(spec: dict[str, Any], *, module_name: str) -> str:
    module = _synapse_module()
    render_fn = getattr(module, "synapse_spec_to_axon_module_text")
    return render_fn(spec, module_name=module_name)


def _ensure_overwrite_allowed(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise typer.BadParameter(
            f"Refusing to overwrite existing file: {path}. Use --force to overwrite."
        )


@app.command("emit")
def emit_generic(
    spec_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a Synapse YAML spec.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination Python file for generated model code.",
    ),
    class_name: str = typer.Option(
        "GeneratedSynapseModel",
        "--class-name",
        help="Name of the generated model class.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Generate standalone PyTorch model code from any Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".py":
        raise typer.BadParameter("Output path must end with .py")

    spec = _load_yaml_mapping(spec_path)
    try:
        source = _emit_model_code(spec, class_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    typer.echo(f"Wrote generated model code to {output_path}")


@app.command("axon-to-synapse")
def axon_to_synapse(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination YAML file for lowered Synapse spec.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main model module name when Axon file contains multiple modules (defaults to last).",
    ),
) -> None:
    """Lower an Axon module into a Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix not in {".yaml", ".yml"}:
        raise typer.BadParameter("Output path must end with .yaml or .yml")

    if isinstance(main_module, OptionInfo):
        main_module = None
    try:
        spec = _parse_axon_to_synapse_spec(axon_path, main_module=main_module)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(OmegaConf.to_yaml(spec, resolve=True), encoding="utf-8")
    typer.echo(f"Wrote Synapse YAML to {output_path}")


@app.command("synapse-to-axon")
def synapse_to_axon(
    spec_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a Synapse YAML spec.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination Axon file.",
    ),
    module_name: str = typer.Option(
        "main",
        "--module-name",
        help="Module name to use in emitted Axon source.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Render an Axon source file from a Synapse YAML spec."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".axon":
        raise typer.BadParameter("Output path must end with .axon")
    if not module_name.isidentifier():
        raise typer.BadParameter(f"Invalid module name: {module_name!r}")

    spec = _load_yaml_mapping(spec_path)
    try:
        text = _render_synapse_to_axon_text(spec, module_name=module_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote Axon source to {output_path}")


@app.command("axon-test")
def axon_test(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    weights: Path = typer.Argument(
        ...,
        help="Path to a .safetensors file or model directory.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    text: list[str] = typer.Option(
        ["The future of AI is", "Hello World"],
        "--text",
        help="Prompt text to complete. Repeat --text for batched prompts.",
    ),
    max_len: int = typer.Option(
        32,
        "--max-len",
        help="Total sequence length target for generation.",
    ),
    hf_model_dir: Path | None = typer.Option(
        None,
        "--hf-model-dir",
        help="HF model directory for AutoModel (defaults to weights directory).",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        help="Tokenizer source override (local path or HF repo id).",
    ),
    class_name: str = typer.Option(
        "AxonGeneratedModel",
        "--class-name",
        help="Generated PyTorch class name.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        help="Floating-point dtype for loaded safetensors parameters (float32/bfloat16/float16).",
    ),
    model_task: str = typer.Option(
        "auto",
        "--model-task",
        help="Model execution task (auto, causal_lm, masked_lm, or seq2seq_lm).",
    ),
    hf_align_bf16_profile: bool = typer.Option(
        False,
        "--hf-align-bf16-profile/--no-hf-align-bf16-profile",
        help="Enable a general HF-BF16 alignment profile (mask, posids, add/linear/norm fp32-accum paths).",
    ),
    hf_align_mask_contract: bool = typer.Option(
        False,
        "--hf-align-mask-contract/--no-hf-align-mask-contract",
        help="When enabled, normalize additive attention masks to HF-like SDPA bool masks.",
    ),
    hf_align_position_ids: bool = typer.Option(
        False,
        "--hf-align-position-ids/--no-hf-align-position-ids",
        help="When enabled, use HF-like padding fill behavior for position_ids.",
    ),
    hf_align_add_fp32_accum: bool = typer.Option(
        False,
        "--hf-align-add-fp32-accum/--no-hf-align-add-fp32-accum",
        help="When enabled, compute low-precision add in fp32 and cast back.",
    ),
    hf_align_linear_fp32_accum: bool = typer.Option(
        False,
        "--hf-align-linear-fp32-accum/--no-hf-align-linear-fp32-accum",
        help="When enabled, compute low-precision linear in fp32 and cast back.",
    ),
    hf_align_norm_fp32: bool = typer.Option(
        False,
        "--hf-align-norm-fp32/--no-hf-align-norm-fp32",
        help="When enabled, run low-precision norm ops through fp32 compute paths.",
    ),
    compile_hf: bool = typer.Option(
        False,
        "--compile-hf/--no-compile-hf",
        help="Compile the HF reference model with torch.compile.",
    ),
    compile_axon: bool = typer.Option(
        False,
        "--compile-axon/--no-compile-axon",
        help="Compile the Axon-derived model with torch.compile.",
    ),
    compile_backend: str | None = typer.Option(
        None,
        "--compile-backend",
        help="Optional torch.compile backend (e.g. inductor).",
    ),
    compile_mode: str | None = typer.Option(
        None,
        "--compile-mode",
        help="Optional torch.compile mode (e.g. default/reduce-overhead/max-autotune).",
    ),
    compile_fullgraph: bool = typer.Option(
        False,
        "--compile-fullgraph/--no-compile-fullgraph",
        help="Set torch.compile(fullgraph=True).",
    ),
    compile_dynamic: bool = typer.Option(
        False,
        "--compile-dynamic/--no-compile-dynamic",
        help="Set torch.compile(dynamic=True).",
    ),
) -> None:
    """Run HF vs Axon-derived model benchmark for an Axon spec + weights."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_test")
    if not weights.exists():
        raise typer.BadParameter(f"Weights path not found: {weights}")
    try:
        run_fn(
            axon_file=axon_path,
            weights=weights,
            device=device,
            text=text,
            max_len=max_len,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            class_name=class_name,
            main_module=main_module,
            dtype=dtype,
            model_task=model_task,
            hf_align_bf16_profile=hf_align_bf16_profile,
            hf_align_mask_contract=hf_align_mask_contract,
            hf_align_position_ids=hf_align_position_ids,
            hf_align_add_fp32_accum=hf_align_add_fp32_accum,
            hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
            hf_align_norm_fp32=hf_align_norm_fp32,
            compile_hf=compile_hf,
            compile_axon=compile_axon,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("axon-benchmark")
def axon_benchmark(
    axon_paths: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="One or more Axon source files.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    text: list[str] = typer.Option(
        ["The future of AI is", "Hello World"],
        "--text",
        help="Prompt text to complete. Repeat --text for batched prompts.",
    ),
    max_len: int = typer.Option(
        32,
        "--max-len",
        help="Total sequence length target for generation.",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        help="Tokenizer source override (local path or HF repo id).",
    ),
    class_name: str = typer.Option(
        "AxonGeneratedModel",
        "--class-name",
        help="Generated PyTorch class name.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        help="Floating-point dtype for loaded safetensors parameters (float32/bfloat16/float16).",
    ),
    model_task: str = typer.Option(
        "auto",
        "--model-task",
        help="Model execution task (auto, causal_lm, masked_lm, or seq2seq_lm).",
    ),
    hf_align_bf16_profile: bool = typer.Option(
        False,
        "--hf-align-bf16-profile/--no-hf-align-bf16-profile",
        help="Enable a general HF-BF16 alignment profile (mask, posids, add/linear/norm fp32-accum paths).",
    ),
    hf_align_mask_contract: bool = typer.Option(
        False,
        "--hf-align-mask-contract/--no-hf-align-mask-contract",
        help="When enabled, normalize additive attention masks to HF-like SDPA bool masks.",
    ),
    hf_align_position_ids: bool = typer.Option(
        False,
        "--hf-align-position-ids/--no-hf-align-position-ids",
        help="When enabled, use HF-like padding fill behavior for position_ids.",
    ),
    hf_align_add_fp32_accum: bool = typer.Option(
        False,
        "--hf-align-add-fp32-accum/--no-hf-align-add-fp32-accum",
        help="When enabled, compute low-precision add in fp32 and cast back.",
    ),
    hf_align_linear_fp32_accum: bool = typer.Option(
        False,
        "--hf-align-linear-fp32-accum/--no-hf-align-linear-fp32-accum",
        help="When enabled, compute low-precision linear in fp32 and cast back.",
    ),
    hf_align_norm_fp32: bool = typer.Option(
        False,
        "--hf-align-norm-fp32/--no-hf-align-norm-fp32",
        help="When enabled, run low-precision norm ops through fp32 compute paths.",
    ),
    compile_hf: bool = typer.Option(
        False,
        "--compile-hf/--no-compile-hf",
        help="Compile the HF reference model with torch.compile.",
    ),
    compile_axon: bool = typer.Option(
        False,
        "--compile-axon/--no-compile-axon",
        help="Compile the Axon-derived model with torch.compile.",
    ),
    compile_backend: str | None = typer.Option(
        None,
        "--compile-backend",
        help="Optional torch.compile backend (e.g. inductor).",
    ),
    compile_mode: str | None = typer.Option(
        None,
        "--compile-mode",
        help="Optional torch.compile mode (e.g. default/reduce-overhead/max-autotune).",
    ),
    compile_fullgraph: bool = typer.Option(
        False,
        "--compile-fullgraph/--no-compile-fullgraph",
        help="Set torch.compile(fullgraph=True).",
    ),
    compile_dynamic: bool = typer.Option(
        False,
        "--compile-dynamic/--no-compile-dynamic",
        help="Set torch.compile(dynamic=True).",
    ),
    table_format: str = typer.Option(
        "markdown",
        "--table-format",
        help="Summary table format (plain/markdown).",
    ),
) -> None:
    """Run benchmark across declared CHECKPOINTS for one or more Axon files."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_benchmark")
    try:
        run_fn(
            axon_files=axon_paths,
            device=device,
            text=text,
            max_len=max_len,
            tokenizer=tokenizer,
            class_name=class_name,
            main_module=main_module,
            dtype=dtype,
            model_task=model_task,
            hf_align_bf16_profile=hf_align_bf16_profile,
            hf_align_mask_contract=hf_align_mask_contract,
            hf_align_position_ids=hf_align_position_ids,
            hf_align_add_fp32_accum=hf_align_add_fp32_accum,
            hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
            hf_align_norm_fp32=hf_align_norm_fp32,
            compile_hf=compile_hf,
            compile_axon=compile_axon,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            table_format=table_format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("axon-op-parity")
def axon_op_parity(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    weights: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to a .safetensors file or a model directory containing safetensors.",
    ),
    hf_model_dir: Path | None = typer.Option(
        None,
        "--hf-model-dir",
        help="HF model directory for AutoModel (defaults to weights directory).",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        help="Tokenizer source override (local path or HF repo id).",
    ),
    text: list[str] = typer.Option(
        ["The future of AI is", "Hello world"],
        "--text",
        help="Prompt text for forward-pass parity. Repeat --text for batched prompts.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    dtypes: list[str] = typer.Option(
        ["float32", "bfloat16", "float16"],
        "--dtype",
        help="Dtypes to sweep (repeat --dtype): float32/bfloat16/float16.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional path to write a full JSON report.",
    ),
) -> None:
    """Run per-op HF-internals vs Synapse parity harness across requested dtypes."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_op_parity")
    try:
        run_fn(
            axon_file=axon_path,
            weights=weights,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            text=text,
            device=device,
            dtypes=dtypes,
            output_json=output_json,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("axon-test-matrix")
def axon_test_matrix(
    examples_dir: Path = typer.Option(
        Path("examples"),
        "--examples-dir",
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory with Axon files (default: examples).",
    ),
    models_dir: Path = typer.Option(
        Path("models"),
        "--models-dir",
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory with model directories (default: models).",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    processes: int = typer.Option(
        1,
        "--processes",
        min=1,
        help=(
            "Number of model-evaluation worker processes to run simultaneously. "
            "When using CUDA with multiple processes, workers are assigned across GPUs."
        ),
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        help="Floating-point dtype for loaded safetensors parameters (float32/bfloat16/float16).",
    ),
    max_len: int = typer.Option(
        32,
        "--max-len",
        help="Total sequence length target for generation.",
    ),
    text: list[str] = typer.Option(
        [],
        "--text",
        help="Prompt text to complete. Repeat --text for batched prompts.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show per-run output from synapse axon-test.",
    ),
    no_capture_output: bool = typer.Option(
        False,
        "--no-capture-output",
        help="Do not capture per-run output; stream synapse axon-test output directly.",
    ),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        file_okay=False,
        dir_okay=True,
        help=(
            "If set, each model worker writes stdout/stderr to a separate log file "
            "under this directory using log-<pid>-<axon>-<model>.txt."
        ),
    ),
    include: list[str] = typer.Option(
        [],
        "--include",
        help=(
            "Only run pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        help=(
            "Exclude pairs matching these selectors (repeatable). "
            "Selectors are model directory names or .axon file names."
        ),
    ),
    min_billions_params: float | None = typer.Option(
        None,
        "--min-billions-params",
        help=(
            "Only run models with estimated parameter count at or above this many "
            "billions of parameters."
        ),
    ),
    max_billions_params: float | None = typer.Option(
        None,
        "--max-billions-params",
        help=(
            "Only run models with estimated parameter count at or below this many "
            "billions of parameters."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only resolve and print matching pairs; do not run tests.",
    ),
    table_format: str = typer.Option(
        "plain",
        "--table-format",
        help="Summary table format (plain/markdown).",
    ),
    compile_hf: bool = typer.Option(
        False,
        "--compile-hf/--no-compile-hf",
        help="Compile the HF reference model with torch.compile.",
    ),
    compile_axon: bool = typer.Option(
        False,
        "--compile-axon/--no-compile-axon",
        help="Compile the Axon-derived model with torch.compile.",
    ),
    compile_backend: str | None = typer.Option(
        None,
        "--compile-backend",
        help="Optional torch.compile backend (e.g. inductor).",
    ),
    compile_mode: str | None = typer.Option(
        None,
        "--compile-mode",
        help="Optional torch.compile mode (e.g. default/reduce-overhead/max-autotune).",
    ),
    compile_fullgraph: bool = typer.Option(
        False,
        "--compile-fullgraph/--no-compile-fullgraph",
        help="Set torch.compile(fullgraph=True).",
    ),
    compile_dynamic: bool = typer.Option(
        False,
        "--compile-dynamic/--no-compile-dynamic",
        help="Set torch.compile(dynamic=True).",
    ),
    model_task: str = typer.Option(
        "auto",
        "--model-task",
        help="Execution task override: auto/causal_lm/masked_lm/seq2seq_lm.",
    ),
) -> None:
    """Run synapse axon-test across matching examples/*.axon and models/* directories."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_test_matrix")
    try:
        exit_code = run_fn(
            examples_dir=examples_dir,
            models_dir=models_dir,
            device=device,
            processes=processes,
            dtype=dtype,
            min_billions_params=min_billions_params,
            max_billions_params=max_billions_params,
            max_len=max_len,
            text=text or None,
            verbose=verbose,
            no_capture_output=no_capture_output,
            log_dir=log_dir,
            dry_run=dry_run,
            table_format=table_format,
            compile_hf=compile_hf,
            compile_axon=compile_axon,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            model_task_override=None if model_task == "auto" else model_task,
            include=include or None,
            exclude=exclude or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    raise typer.Exit(code=int(exit_code))


@app.command("axon-test-log")
def axon_test_log(
    log_dir: Path = typer.Argument(
        Path("log"),
        file_okay=False,
        dir_okay=True,
        readable=False,
        help="Directory containing parent-<pid>.txt and log-<pid>-<axon>-<model>.txt files.",
    ),
    all_runs: bool = typer.Option(
        False,
        "--all",
        help=(
            "Parse all parent-tracked runs in the directory, but keep only the most recent "
            "result for each (axon, model dir) pair."
        ),
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help=(
            "Delete all parent-*.txt and log-*.txt files except the most recent parent run "
            "and the worker logs it references, then print the table."
        ),
    ),
) -> None:
    """Parse axon-test-matrix logs and print a markdown summary table."""
    module = _synapse_module()
    render_fn = getattr(module, "render_axon_test_log")
    try:
        output = render_fn(log_dir, all_runs=all_runs, prune=prune)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(output)


@app.command("axon-materialize")
def axon_materialize(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a config-driven Axon source file to materialize.",
    ),
    checkpoint: list[str] = typer.Option(
        [],
        "--checkpoint",
        help=(
            "Checkpoint repo id to materialize. Repeatable. If omitted, the materializer uses "
            "its default checkpoint set."
        ),
    ),
    models_root: Path = typer.Option(
        Path("models"),
        "--models-root",
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory containing local model checkpoint directories.",
    ),
) -> None:
    """Materialize checkpoint-specific Axon files with config values baked in."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_materialize")
    try:
        written = run_fn(
            axon_path=axon_path,
            checkpoints=checkpoint or None,
            models_root=models_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for path in written:
        typer.echo(path)


@app.command("axon-visualize")
def axon_visualize(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    output_path: Path = typer.Argument(
        ...,
        help="Destination Graphviz DOT file.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main model module name when Axon file contains multiple modules (defaults to last).",
    ),
    control_flow: bool = typer.Option(
        True,
        "--control-flow/--no-control-flow",
        help="Show dashed gray control-flow edges between ops.",
    ),
    direction: str = typer.Option(
        "top-down",
        "--direction",
        help="Graph layout direction: top-down, bottom-up, left-right, or right-left.",
    ),
) -> None:
    """Lower an Axon program and write a DOT graph of blocks + ops + variable-flow edges."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".dot":
        raise typer.BadParameter("Output path must end with .dot")
    if isinstance(main_module, OptionInfo):
        main_module = None
    if isinstance(control_flow, OptionInfo):
        control_flow = True
    if isinstance(direction, OptionInfo):
        direction = "top-down"

    module = _synapse_module()
    run_fn = getattr(module, "run_axon_visualize")
    try:
        written = run_fn(
            axon_file=axon_path,
            output_path=output_path,
            main_module=main_module,
            show_control_flow=control_flow,
            direction=direction,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote Axon graph visualization to {written}")


__all__ = [
    "app",
    "emit_generic",
    "axon_to_synapse",
    "synapse_to_axon",
    "axon_test",
    "axon_op_parity",
    "axon_test_matrix",
    "axon_visualize",
]
