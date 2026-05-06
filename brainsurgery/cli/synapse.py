from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf
from typer.models import OptionInfo

from .synapse_materialize import run_axon_materialize_workflow

app = typer.Typer(help="Synapse tooling.")


def _synapse_module() -> Any:
    return importlib.import_module("brainsurgery.synapse")


def _axon_module() -> Any:
    return importlib.import_module("brainsurgery.synapse.axon")


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


def _resolve_axon_to_text(axon_path: Path, *, strict: bool = False) -> str:
    module = _synapse_module()
    resolve_fn = getattr(module, "resolve_axon_program_to_source")
    return resolve_fn(axon_path, strict=strict)


def _resolve_axon_program(axon_path: Path, *, strict: bool = False) -> Any:
    module = _synapse_module()
    resolve_fn = getattr(module, "resolve_axon_program_from_path")
    return resolve_fn(axon_path, strict=strict)


def _render_resolved_axon_program(program: Any) -> str:
    module = _synapse_module()
    render_fn = getattr(module, "render_axon_file")
    return render_fn(program.ast)


_AXON_DUMP_STAGES = {
    "parse",
    "resolve",
    "normalize",
    "flatten",
    "typecheck",
    "backend-required",
    "optimize",
    "canonicalize",
    "final",
}


def _dump_axon_stage_to_text(
    axon_path: Path,
    *,
    stage: str,
    main_module: str | None,
    strict: bool,
    backend_required: bool,
    optimize: bool,
    canonicalize: bool,
    show_types: bool,
) -> str:
    module = _axon_module()
    parse_fn = getattr(module, "parse_axon_program_from_path")
    resolve_fn = getattr(module, "resolve_axon_program_from_path")
    normalize_fn = getattr(module, "normalize_closed_axon_file")
    flatten_fn = getattr(module, "flatten_closed_axon_file")
    typecheck_fn = getattr(module, "typecheck_flat_axon_file")
    backend_required_fn = getattr(module, "normalize_backend_required_flat_typed_axon_file")
    optimize_fn = getattr(module, "optimize_flat_typed_axon_file")
    canonicalize_fn = getattr(module, "canonicalize_typed_axon_file")
    render_fn = getattr(module, "render_axon_file")

    if stage not in _AXON_DUMP_STAGES:
        allowed = ", ".join(sorted(_AXON_DUMP_STAGES))
        raise typer.BadParameter(f"Unknown stage {stage!r}. Expected one of: {allowed}")
    if stage == "backend-required" and not backend_required:
        raise typer.BadParameter("--stage backend-required requires --backend-required")
    if stage == "optimize" and not optimize:
        raise typer.BadParameter("--stage optimize requires --optimize")
    if stage == "canonicalize" and not canonicalize:
        raise typer.BadParameter("--stage canonicalize requires --canonicalize")

    if stage == "parse":
        return render_fn(parse_fn(axon_path), show_types=show_types)

    report = resolve_fn(axon_path, strict=strict)
    program = report.ast
    if stage == "resolve":
        return render_fn(program, show_types=show_types)

    program = normalize_fn(program, main_module=main_module)
    if stage == "normalize":
        return render_fn(program, show_types=show_types)

    program = flatten_fn(program, main_module=main_module)
    if stage == "flatten":
        return render_fn(program, show_types=show_types)

    program = typecheck_fn(program, main_module=main_module)
    if stage == "typecheck":
        return render_fn(program, show_types=show_types)

    if stage == "backend-required":
        program = backend_required_fn(program, main_module=main_module)
        return render_fn(program, show_types=show_types)

    if stage == "optimize" or (stage == "final" and optimize):
        program = optimize_fn(program, main_module=main_module)
        if stage == "optimize":
            return render_fn(program, show_types=show_types)

    if stage == "final" and not optimize and backend_required:
        program = backend_required_fn(program, main_module=main_module)

    if stage == "canonicalize" or (stage == "final" and canonicalize):
        program = canonicalize_fn(program, main_module=main_module)

    return render_fn(program, show_types=show_types)


def _build_pipeline_plan_for_axon(
    axon_path: Path, *, device: str = "cuda", main_module: str | None = None
) -> Any:
    module = _synapse_module()
    parse_fn = getattr(module, "parse_axon_program_from_path")
    lower_fn = getattr(module, "lower_axon_program_to_synapse_spec")
    plan_fn = getattr(module, "build_pipeline_plan")
    parsed = parse_fn(axon_path)
    spec = lower_fn(parsed, main_module=main_module)
    return plan_fn(spec, requested_device=device)


def _emit_pipeline_stage_codes_for_axon(
    axon_path: Path,
    *,
    device: str = "cuda",
    main_module: str | None = None,
    class_name_prefix: str = "GeneratedPipelineStage",
) -> tuple[Any, tuple[str, ...]]:
    module = _synapse_module()
    parse_fn = getattr(module, "parse_axon_program_from_path")
    lower_fn = getattr(module, "lower_axon_program_to_synapse_spec")
    emit_fn = getattr(module, "emit_pipeline_stage_codes_from_synapse_spec")
    parsed = parse_fn(axon_path)
    spec = lower_fn(parsed, main_module=main_module)
    return emit_fn(spec, requested_device=device, class_name_prefix=class_name_prefix)


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


@app.command("axon-resolve")
def axon_resolve(
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
        help="Destination Axon file with imports resolved away.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail when the resolver emits warnings.",
    ),
) -> None:
    """Resolve imports into one Axon file without import statements."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".axon":
        raise typer.BadParameter("Output path must end with .axon")

    try:
        program = _resolve_axon_program(axon_path, strict=strict)
        text = _render_resolved_axon_program(program)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    for diagnostic in program.diagnostics:
        level = diagnostic.level.upper()
        prefix = f"{diagnostic.file_path}: " if diagnostic.file_path is not None else ""
        typer.echo(f"{level}: {prefix}{diagnostic.message}", err=True)
    typer.echo(f"Wrote resolved Axon source to {output_path}")


@app.command("axon-stage-dump")
def axon_stage_dump(
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
        help="Destination Axon file for the selected pipeline stage.",
    ),
    stage: str = typer.Option(
        "final",
        "--stage",
        "--target-stage",
        help=(
            "Stage to dump: parse, resolve, normalize, flatten, typecheck, "
            "backend-required, optimize, canonicalize, or final."
        ),
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail when the resolver emits warnings.",
    ),
    backend_required: bool = typer.Option(
        True,
        "--backend-required/--no-backend-required",
        help="For --stage final --no-optimize, run backend-required rewrites after typecheck.",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize/--no-optimize",
        help="For --stage final, run the optimizer after typecheck/backend-required.",
    ),
    canonicalize: bool = typer.Option(
        False,
        "--canonicalize/--no-canonicalize",
        help="For --stage final, run canonicalization last.",
    ),
    show_types: bool = typer.Option(
        False,
        "--show-types/--no-show-types",
        help="Render inferred type annotations when available.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Render the Axon program after a selected frontend/pipeline stage."""
    _ensure_overwrite_allowed(output_path, force=force)
    if output_path.suffix != ".axon":
        raise typer.BadParameter("Output path must end with .axon")
    if isinstance(main_module, OptionInfo):
        main_module = None

    try:
        text = _dump_axon_stage_to_text(
            axon_path,
            stage=stage,
            main_module=main_module,
            strict=strict,
            backend_required=backend_required,
            optimize=optimize,
            canonicalize=canonicalize,
            show_types=show_types,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {stage} Axon source to {output_path}")


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
    trace_layers: bool = typer.Option(
        False,
        "--trace-layers/--no-trace-layers",
        help="Compare traced HF and Axon layer inputs/outputs when supported.",
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
    hf_strict_dtype: bool = typer.Option(
        False,
        "--hf-strict-dtype/--no-hf-strict-dtype",
        help="Force HF floating tensors to exactly match --dtype and disable HF quantization overrides when needed.",
    ),
    oom_cpu_fallback: bool = typer.Option(
        True,
        "--oom-cpu-fallback/--no-oom-cpu-fallback",
        help="On CUDA OOM, retry HF/Axon on CPU (disable to fail fast on OOM).",
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize/--no-optimize",
        help="Enable the Axon optimizer before lowering.",
    ),
    canonicalize: bool = typer.Option(
        False,
        "--canonicalize/--no-canonicalize",
        help="Enable Axon canonicalization before codegen2/runtime2 graph lowering.",
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
            trace_layers=trace_layers,
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
            hf_strict_dtype=hf_strict_dtype,
            optimize=optimize,
            canonicalize=canonicalize,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("axon-pipeline-plan")
def axon_pipeline_plan(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Pipeline device target (cuda or explicit like cuda:0).",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
) -> None:
    """Show the inferred pipeline layer split for an Axon model."""
    if isinstance(main_module, OptionInfo):
        main_module = None
    try:
        plan = _build_pipeline_plan_for_axon(axon_path, device=device, main_module=main_module)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"layers_var={plan.layers_var}")
    typer.echo(f"total_layers={plan.total_layers}")
    for stage in plan.stages:
        typer.echo(
            f"stage={stage.index} device={stage.device} layers=[{stage.layer_start},{stage.layer_stop})"
        )


@app.command("axon-pipeline-emit")
def axon_pipeline_emit(
    axon_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axon source file.",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Destination directory for emitted per-stage Python files.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Pipeline device target (cuda or explicit like cuda:0).",
    ),
    class_name_prefix: str = typer.Option(
        "GeneratedPipelineStage",
        "--class-name-prefix",
        help="Python class-name prefix for emitted stages.",
    ),
    main_module: str | None = typer.Option(
        None,
        "--main-module",
        help="Main Axon module name (defaults to last module in file).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output files if they already exist.",
    ),
) -> None:
    """Emit separate Python modules for each inferred pipeline stage."""
    if isinstance(main_module, OptionInfo):
        main_module = None
    try:
        plan, codes = _emit_pipeline_stage_codes_for_axon(
            axon_path,
            device=device,
            main_module=main_module,
            class_name_prefix=class_name_prefix,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage, code in zip(plan.stages, codes, strict=True):
        output_path = output_dir / f"stage_{stage.index}.py"
        _ensure_overwrite_allowed(output_path, force=force)
        output_path.write_text(code, encoding="utf-8")
        typer.echo(
            f"wrote {output_path} for {stage.device} layers=[{stage.layer_start},{stage.layer_stop})"
        )


@app.command("axon-benchmark")
def axon_benchmark(
    axon_paths: list[Path] = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="One or more Axon source files or directories to recurse for .axon files.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    processes: int = typer.Option(
        1,
        "--processes",
        help="Number of worker processes.",
    ),
    checkpoints: list[str] = typer.Option(
        [],
        "--checkpoint",
        help="Benchmark only the specified declared checkpoint ID(s). Repeat --checkpoint to select multiple.",
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        help=(
            "Exclude discovered .axon files matching these selectors (repeatable). "
            "Selectors are substring-matched against the file path, name, and stem."
        ),
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
    trace_layers: bool = typer.Option(
        False,
        "--trace-layers/--no-trace-layers",
        help="Compare traced HF and Axon layer inputs/outputs when supported.",
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
    axon_backend: str = typer.Option(
        "codegen",
        "--axon-backend",
        help="Axon execution backend (codegen, codegen2, runtime, runtime2, or pipeline).",
    ),
    axon_typechecker: str = typer.Option(
        "typecheck2",
        "--axon-typechecker",
        help="Axon typechecker for codegen2/runtime2 graph lowering (typecheck or typecheck2).",
    ),
    pipeline_parallel_size: int | None = typer.Option(
        None,
        "--pipeline-parallel-size",
        "--pp",
        help=(
            "Pipeline stages per worker when --axon-backend pipeline is used. "
            "With --processes > 1, each worker gets pp GPUs via CUDA_VISIBLE_DEVICES partitioning."
        ),
    ),
    optimize: bool = typer.Option(
        False,
        "--optimize/--no-optimize",
        help="Enable the Axon optimizer before lowering.",
    ),
    canonicalize: bool = typer.Option(
        False,
        "--canonicalize/--no-canonicalize",
        help="Enable Axon canonicalization before codegen2/runtime2 graph lowering.",
    ),
    skip_hf: bool = typer.Option(
        False,
        "--skip-hf/--no-skip-hf",
        help="Skip the HF reference side and run only AxonDerived. Useful for large pipeline experiments.",
    ),
    hf_strict_dtype: bool = typer.Option(
        False,
        "--hf-strict-dtype/--no-hf-strict-dtype",
        help="Force HF floating tensors to exactly match --dtype and disable HF quantization overrides when needed.",
    ),
    oom_cpu_fallback: bool = typer.Option(
        True,
        "--oom-cpu-fallback/--no-oom-cpu-fallback",
        help="On CUDA OOM, retry HF/Axon on CPU (disable to fail fast on OOM).",
    ),
    table_format: str = typer.Option(
        "markdown",
        "--table-format",
        help="Summary table format (plain/markdown/html).",
    ),
    log_dir: Path | None = typer.Option(
        None,
        "--log-dir",
        help="Optional directory for per-worker logs.",
    ),
    stream_csv: Path | None = typer.Option(
        None,
        "--stream-csv",
        help="Optional CSV file to append unsorted completed benchmark rows to as they finish.",
    ),
    min_billion_parameters: float | None = typer.Option(
        None,
        "--min-billion-parameters",
        help="Only run models with estimated parameter count at or above this many billions.",
    ),
    max_billion_parameters: float | None = typer.Option(
        None,
        "--max-billion-parameters",
        help="Only run models with estimated parameter count at or below this many billions.",
    ),
    debug_errors: bool = typer.Option(
        False,
        "--debug-errors/--no-debug-errors",
        help="Print full traceback details for benchmark pair failures.",
    ),
) -> None:
    """Run benchmark across declared CHECKPOINTS for one or more Axon files."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_benchmark")
    try:
        run_fn(
            axon_files=axon_paths,
            checkpoints=checkpoints,
            exclude=exclude,
            device=device,
            processes=processes,
            text=text,
            max_len=max_len,
            tokenizer=tokenizer,
            class_name=class_name,
            dtype=dtype,
            model_task=model_task,
            trace_layers=trace_layers,
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
            axon_backend=axon_backend,
            axon_typechecker=axon_typechecker,
            pipeline_parallel_size=pipeline_parallel_size,
            optimize=optimize,
            canonicalize=canonicalize,
            skip_hf=skip_hf,
            hf_strict_dtype=hf_strict_dtype,
            oom_cpu_fallback=oom_cpu_fallback,
            table_format=table_format,
            log_dir=log_dir,
            stream_csv=stream_csv,
            debug_errors=debug_errors,
            min_billion_parameters=min_billion_parameters,
            max_billion_parameters=max_billion_parameters,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("axon-benchmark-render")
def axon_benchmark_render(
    csv_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV file previously written by axon-benchmark --stream-csv.",
    ),
    table_format: str = typer.Option(
        "markdown",
        "--table-format",
        help="Rendered table format (plain/markdown/html).",
    ),
) -> None:
    """Render a streamed axon-benchmark CSV file as a sorted table."""
    module = _synapse_module()
    render_fn = getattr(module, "render_axon_benchmark_csv")
    try:
        typer.echo(render_fn(csv_path=csv_path, table_format=table_format))
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


@app.command("axon-layer-op-parity")
def axon_layer_op_parity(
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
    layer_index: int = typer.Option(
        ...,
        "--layer-index",
        min=0,
        help="Layer index to compare.",
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
        ["Hello world"],
        "--text",
        help="Prompt text for forward-pass parity. Repeat --text for batched prompts.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device (cpu/auto/cuda/mps or explicit like cuda:0).",
    ),
    dtype: str = typer.Option(
        "float32",
        "--dtype",
        help="Dtype to run (float32/bfloat16/float16).",
    ),
    max_len: int = typer.Option(
        32,
        "--max-len",
        help="Total sequence length target.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional path to write the full JSON report.",
    ),
) -> None:
    """Run layer-scoped HF-internals vs Synapse parity harness."""
    module = _synapse_module()
    run_fn = getattr(module, "run_axon_layer_op_parity")
    try:
        result = run_fn(
            axon_file=axon_path,
            weights=weights,
            layer_index=layer_index,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            text=text,
            device=device,
            dtype=dtype,
            max_len=max_len,
        )
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            typer.echo(f"[axon-layer-op-parity] wrote report: {output_json}")
        by_op = result.get("by_op", {})
        for kind in sorted(by_op):
            stats = by_op[kind]
            typer.echo(
                f"[axon-layer-op-parity] {kind} max_abs={stats.get('max_abs')} "
                f"mean_abs={stats.get('mean_abs')} matched={stats.get('matched')}"
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


@app.command("axon-benchmark-log")
def axon_benchmark_log(
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
    table_format: str = typer.Option(
        "markdown",
        "--table-format",
        help="Rendered table format (plain/markdown/html).",
    ),
) -> None:
    """Parse axon-benchmark or axon-test-matrix logs and print a summary table."""
    module = _synapse_module()
    render_fn = getattr(module, "render_axon_benchmark_log")
    try:
        output = render_fn(log_dir, all_runs=all_runs, prune=prune, table_format=table_format)
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
    try:
        written = run_axon_materialize_workflow(
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
