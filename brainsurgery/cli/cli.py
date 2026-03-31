import logging
from pathlib import Path
from typing import Any

import typer
from typer.models import ArgumentInfo, OptionInfo

from ..engine import (
    GpuCacheConfig,
    ProviderError,
    RuntimeFlagLifecycleScope,
    apply_log_level,
    compile_plan,
    create_state_dict_provider,
    get_runtime_flags,
    list_model_aliases,
    normalize_raw_plan,
    reset_runtime_flags_for_scope,
    use_output_emitter,
    wrap_provider_with_gpu_cache,
)
from .config import _load_cli_config
from .history import _configure_history
from .interactive import _prompt_interactive_transform
from .summary import _write_executed_plan_summary

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")

app = typer.Typer(help="Brain surgery CLI.")


def configure_logging(log_level: str) -> None:
    try:
        apply_log_level(log_level)
    except ValueError:
        raise typer.BadParameter("log-level must be one of: critical, debug, error, info, warning")


def _resolve_typer_default(value: Any) -> Any:
    if isinstance(value, (OptionInfo, ArgumentInfo)):
        return value.default
    return value


def _execute_configured_transforms(
    *,
    surgery_plan: Any,
    state_dict_provider: Any,
) -> bool:
    should_continue = surgery_plan.execute_pending(
        state_dict_provider,
        interactive=False,
    )
    return should_continue


def _run_interactive_session(
    *,
    surgery_plan: Any,
    state_dict_provider: Any,
) -> bool:
    while True:
        extra_specs = _prompt_interactive_transform(state_dict_provider=state_dict_provider)
        if extra_specs is None:
            logger.info("Interactive session complete")
            return True

        before_count = len(surgery_plan.steps)
        try:
            surgery_plan.append_raw_transforms(extra_specs)
            surgery_plan.compile_pending(
                extra_known_models=set(list_model_aliases(state_dict_provider)),
            )
        except Exception as exc:
            del surgery_plan.steps[before_count:]
            logger.error("Could not compile interactive transform(s): %s", exc)
            continue

        should_continue = surgery_plan.execute_pending(
            state_dict_provider,
            interactive=True,
        )
        if not should_continue:
            logger.info("Leaving interactive mode")
            return False


@app.callback(invoke_without_command=True)
def run(
    config_items: list[str] = typer.Argument(
        None,
        help=(
            "YAML plan fragments and/or key=value overrides. YAML files are loaded and "
            "deep-merged in order; overrides are applied last. Typically used to define "
            "inputs, transforms, and output."
        ),
    ),
    shard_size: str = typer.Option(
        "5GB",
        help="Default shard size when writing directory outputs (e.g. safetensors shards). Ignored for single-file outputs.",
    ),
    num_workers: int = typer.Option(
        8,
        help="Maximum parallel workers for loading and saving tensors. Higher values improve I/O throughput but increase memory pressure.",
    ),
    provider: str = typer.Option(
        "inmemory",
        help="State-dict backend. 'inmemory' loads tensors into RAM; 'arena' memory-maps tensors to disk for large models.",
    ),
    arena_root: Path = typer.Option(
        Path(".brainsurgery"),
        help="Directory for arena storage when using --provider arena (memory-mapped tensor backing).",
    ),
    arena_segment_size: str = typer.Option(
        "1GB",
        help="Segment size for arena storage (e.g. 1GB). Larger segments reduce fragmentation but use more disk.",
    ),
    interactive: bool = typer.Option(
        False,
        "-i",
        "--interactive",
        help="Run configured transforms, then enter an interactive prompt to execute additional transforms incrementally.",
    ),
    summarize: bool = typer.Option(
        True,
        "-s",
        "--summarize/--no-summarize",
        help="Write a YAML summary of the transforms actually executed (after overrides and interactive edits).",
    ),
    summarize_path: Path | None = typer.Option(
        None,
        help="Destination for the executed-plan summary. Defaults to stdout if not set.",
    ),
    summary_mode: str = typer.Option(
        "raw",
        "--summary-mode",
        help="Summary representation mode: 'raw' (original transform payloads) or 'resolve' (compiled/canonical plan).",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Logging verbosity (debug, info, warning, error, critical).",
    ),
    gpu_cache_device: str | None = typer.Option(
        None,
        "--gpu-cache-device",
        help=(
            "Enable GPU tensor cache for state dicts on this device (e.g. mps, cuda, cuda:0). "
            "When omitted, GPU caching is disabled."
        ),
    ),
    gpu_cache_fraction: float = typer.Option(
        0.8,
        "--gpu-cache-fraction",
        help=(
            "Fraction of detected device memory used as cache budget when --gpu-cache-bytes "
            "is not set."
        ),
    ),
    gpu_cache_bytes: int | None = typer.Option(
        None,
        "--gpu-cache-bytes",
        help=("Explicit GPU cache budget in bytes. Overrides --gpu-cache-fraction when provided."),
    ),
    gpu_cache_non_blocking: bool = typer.Option(
        False,
        "--gpu-cache-non-blocking/--no-gpu-cache-non-blocking",
        help="Use non-blocking device transfers in GPU cache operations when possible.",
    ),
    gpu_cache_debug: bool = typer.Option(
        False,
        "--gpu-cache-debug/--no-gpu-cache-debug",
        help="Emit detailed GPU cache lifecycle logs (hits/misses, transfers, dirty/write-back, evictions).",
    ),
) -> None:
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    config_items = _resolve_typer_default(config_items)
    shard_size = _resolve_typer_default(shard_size)
    num_workers = _resolve_typer_default(num_workers)
    provider = _resolve_typer_default(provider)
    arena_root = _resolve_typer_default(arena_root)
    arena_segment_size = _resolve_typer_default(arena_segment_size)
    interactive = _resolve_typer_default(interactive)
    summarize = _resolve_typer_default(summarize)
    summarize_path = _resolve_typer_default(summarize_path)
    summary_mode = _resolve_typer_default(summary_mode)
    log_level = _resolve_typer_default(log_level)
    gpu_cache_device = _resolve_typer_default(gpu_cache_device)
    gpu_cache_fraction = _resolve_typer_default(gpu_cache_fraction)
    gpu_cache_bytes = _resolve_typer_default(gpu_cache_bytes)
    gpu_cache_non_blocking = _resolve_typer_default(gpu_cache_non_blocking)
    gpu_cache_debug = _resolve_typer_default(gpu_cache_debug)
    config_items = config_items or []

    configure_logging(log_level)
    _configure_history()
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)

    raw_plan = _load_cli_config(config_items)
    planned_raw = normalize_raw_plan(raw_plan)

    logger.info(
        "Scrubbing in. Surgical plan assembled from %d config item(s)",
        len(config_items),
    )
    surgery_plan = compile_plan(planned_raw)
    logger.info(
        "Surgical plan ready: %d brain(s) prepped, %d procedure(s) scheduled, preservation %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path if surgery_plan.output else None,
    )

    try:
        state_dict_provider = create_state_dict_provider(
            provider=provider,
            model_paths=surgery_plan.inputs,
            max_io_workers=num_workers,
            arena_root=arena_root,
            arena_segment_size=arena_segment_size,
        )
        if gpu_cache_device is not None:
            cache_config = GpuCacheConfig(
                device=gpu_cache_device,
                max_cache_bytes=gpu_cache_bytes,
                memory_fraction=gpu_cache_fraction,
                non_blocking=gpu_cache_non_blocking,
                debug=gpu_cache_debug,
            )
            state_dict_provider = wrap_provider_with_gpu_cache(
                state_dict_provider,
                cache_config=cache_config,
            )
            logger.info(
                "GPU state-dict cache enabled on device=%s budget=%s fraction=%.3f non_blocking=%s debug=%s",
                gpu_cache_device,
                gpu_cache_bytes if gpu_cache_bytes is not None else "auto",
                gpu_cache_fraction,
                gpu_cache_non_blocking,
                gpu_cache_debug,
            )
    except ProviderError as exc:
        raise typer.BadParameter(str(exc)) from exc

    written_path: str | Path | None = None

    try:
        with use_output_emitter(typer.echo):
            should_continue = _execute_configured_transforms(
                surgery_plan=surgery_plan,
                state_dict_provider=state_dict_provider,
            )

            if should_continue and interactive:
                logger.info("Entering interactive mode after configured procedures")
                should_continue = _run_interactive_session(
                    surgery_plan=surgery_plan,
                    state_dict_provider=state_dict_provider,
                )

            if surgery_plan.output is None:
                logger.info("No preservation requested; concluding operation without closure")
            elif get_runtime_flags().dry_run:
                logger.info("Dry-run enabled; skipping output save")
            else:
                written_path = state_dict_provider.save_output(
                    surgery_plan,
                    default_shard_size=shard_size,
                    max_io_workers=num_workers,
                )
                logger.info("Operation complete. Brain preserved at %s", written_path)

            if summarize:
                if get_runtime_flags().dry_run and summarize_path is not None:
                    logger.info(
                        "Dry-run enabled; skipping summary file write to %s", summarize_path
                    )
                else:
                    _write_executed_plan_summary(
                        plan=surgery_plan,
                        destination=summarize_path,
                        mode=summary_mode,
                    )

    finally:
        state_dict_provider.close()


if __name__ == "__main__":
    app()


__all__ = ["app", "configure_logging", "run", "logger"]
