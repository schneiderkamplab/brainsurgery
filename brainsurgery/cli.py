from __future__ import annotations

import logging
from pathlib import Path

import typer

from .arena import ArenaError, SegmentedFileBackedArena
from .model import parse_shard_size
from .plan import load_plan
from .providers import ArenaStateDictProvider, InMemoryStateDictProvider
from .transform import apply_transform

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")

app = typer.Typer(help="Brain surgery CLI.")


@app.command()
def run(
    plan: Path,
    shard_size: str = typer.Option("5GB", help="Default shard size for directory outputs"),
    max_io_workers: int = typer.Option(8, help="Max parallel I/O workers"),
    provider: str = typer.Option("inmemory", help="State-dict provider: inmemory or arena"),
    arena_root: Path = typer.Option(
        Path(".brainsurgery"),
        help="Arena directory when using the arena provider",
    ),
    arena_segment_size: str = typer.Option(
        "1GB",
        help="Arena segment size, e.g. 1GB, 4GB, 512MB",
    ),
):
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    logger.info("Scrubbing in with surgical plan %s", plan)
    surgery_plan = load_plan(plan)
    logger.info(
        "Plan loaded: %d input brains, %d transform(s), output path %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path,
    )

    provider_name = provider.strip().lower()

    try:
        if provider_name == "inmemory":
            state_dict_provider = InMemoryStateDictProvider(
                surgery_plan.inputs,
                max_io_workers=max_io_workers,
            )
        elif provider_name == "arena":
            segment_size_bytes = parse_shard_size(arena_segment_size)
            if segment_size_bytes is None:
                raise typer.BadParameter("arena-segment-size must not be 'none'")

            arena = SegmentedFileBackedArena(
                arena_root,
                segment_size_bytes=segment_size_bytes,
            )
            state_dict_provider = ArenaStateDictProvider(
                surgery_plan.inputs,
                arena=arena,
                max_io_workers=max_io_workers,
            )
        else:
            raise typer.BadParameter("provider must be either 'inmemory' or 'arena'")
    except ArenaError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        for transform_index, transform in enumerate(surgery_plan.transforms, start=1):
            logger.info(
                "Transform %d/%d: preparing %s",
                transform_index,
                len(surgery_plan.transforms),
                type(transform.spec).__name__,
            )
            transform_result = apply_transform(transform, state_dict_provider)
            logger.info(
                "Transform %d/%d: %s complete, %d target(s) affected",
                transform_index,
                len(surgery_plan.transforms),
                transform_result.name,
                transform_result.count,
            )

        written_path = state_dict_provider.save_output(
            surgery_plan,
            default_shard_size=shard_size,
            max_io_workers=max_io_workers,
        )
        typer.echo(f"Wrote output checkpoint to {written_path}")
    finally:
        state_dict_provider.close()


if __name__ == "__main__":
    app()
