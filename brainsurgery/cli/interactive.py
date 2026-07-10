import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from ..engine import normalize_transform_specs
from .complete import (
    _collect_completion_candidates,
    _collect_payload_candidates,
    _infer_active_transform,
    _is_top_level_completion_position,
    _list_model_aliases,
    _match_payload_candidates,
)
from .complete import (
    _configure_readline_completion_bindings as _configure_readline_completion_bindings_impl,
)
from .history import _add_history_entry
from .oly import _parse_oly_line
from .parse import _parse_transform_block

logger = logging.getLogger("brainsurgery")
console = Console()

readline: Any | None
try:
    import readline as _readline

    readline = _readline
except ImportError:  # pragma: no cover
    readline = None

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _configure_readline_completion_bindings() -> None:
    _configure_readline_completion_bindings_impl(readline)


def _readline_safe_prompt(prompt_text: str) -> str:
    if readline is None:
        return prompt_text
    return _ANSI_ESCAPE_RE.sub(lambda match: f"\001{match.group(0)}\002", prompt_text)


@contextmanager
def _interactive_completion(
    *,
    top_level_candidates: list[str],
    lines: list[str],
    state_dict_provider: Any | None,
) -> Iterator[None]:
    if readline is None:
        yield
        return

    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    matches: list[str] = []

    def completer(text: str, state: int) -> str | None:
        nonlocal matches
        if state == 0:
            try:
                line_buffer = readline.get_line_buffer()
                begidx = readline.get_begidx()
                get_endidx = getattr(readline, "get_endidx", None)
                if callable(get_endidx):
                    endidx = get_endidx()
                else:
                    endidx = begidx + len(text)
            except Exception:
                line_buffer = ""
                begidx = 0
                endidx = 0

            if _is_top_level_completion_position(line_buffer, begidx):
                matches = [
                    candidate for candidate in top_level_candidates if candidate.startswith(text)
                ]
                if not matches and ":" in line_buffer:
                    active_transform = _infer_active_transform(lines, line_buffer)
                    payload_candidates = _collect_payload_candidates(
                        active_transform=active_transform,
                        state_dict_provider=state_dict_provider,
                        line_buffer=line_buffer,
                    )
                    matches = _match_payload_candidates(
                        text=text,
                        line_buffer=line_buffer,
                        begidx=begidx,
                        endidx=endidx,
                        payload_candidates=payload_candidates,
                        active_transform=active_transform,
                        model_aliases=sorted(_list_model_aliases(state_dict_provider)),
                    )
            else:
                active_transform = _infer_active_transform(lines, line_buffer)
                payload_candidates = _collect_payload_candidates(
                    active_transform=active_transform,
                    state_dict_provider=state_dict_provider,
                    line_buffer=line_buffer,
                )
                matches = _match_payload_candidates(
                    text=text,
                    line_buffer=line_buffer,
                    begidx=begidx,
                    endidx=endidx,
                    payload_candidates=payload_candidates,
                    active_transform=active_transform,
                    model_aliases=sorted(_list_model_aliases(state_dict_provider)),
                )
        if state < len(matches):
            return matches[state]
        return None

    _configure_readline_completion_bindings()

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(completer)
    except Exception:
        logger.debug("Could not configure readline completion", exc_info=True)

    try:
        yield
    finally:
        try:
            readline.set_completer(previous_completer)
            readline.set_completer_delims(previous_delims)
        except Exception:
            logger.debug("Could not restore readline completion", exc_info=True)


def _prompt_interactive_transform(
    state_dict_provider: Any | None = None,
) -> list[dict[str, Any]] | None:
    console.print()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Enter one transform as YAML or OLY, or a YAML list of transforms.",
                    "Finish input with an empty line.",
                    "Tab completion: top-level transforms, then payload keys/aliases/tensors/syntax.",
                    "Special transforms: [cyan]help[/cyan], [cyan]exit[/cyan]",
                    "YAML: [dim]copy: { from: ln_f.weight, to: ln_f_copy.weight }[/dim]",
                    "OLY:  [dim]copy: from: ln_f.weight, to: ln_f_copy.weight[/dim]",
                ]
            ),
            title="Interactive mode",
            border_style="cyan",
        )
    )

    while True:
        lines: list[str] = []
        prompt = _readline_safe_prompt(
            typer.style("brainsurgery> ", fg=typer.colors.CYAN, bold=True)
        )
        continuation = _readline_safe_prompt(typer.style("... ", fg=typer.colors.BRIGHT_BLACK))
        top_level_candidates = _collect_completion_candidates(state_dict_provider)

        with _interactive_completion(
            top_level_candidates=top_level_candidates,
            lines=lines,
            state_dict_provider=state_dict_provider,
        ):
            while True:
                try:
                    line = input(prompt)
                except KeyboardInterrupt:
                    console.print()
                    console.print("KeyboardInterrupt")
                    break
                except EOFError:
                    return None

                if line.strip() == "":
                    if not lines:
                        continue

                    block = "\n".join(lines)
                    try:
                        parsed = _parse_transform_block(block)
                        _add_history_entry(block)
                        return parsed
                    except ValueError as exc:
                        logger.error("Interactive transform rejected: %s", exc)
                        console.print(f"[red]Rejected:[/red] {exc}")
                        console.print("[yellow]Try again.[/yellow]")
                        break

                # Execute complete single-line OLY immediately on enter.
                if not lines:
                    text = line.strip()
                    if text:
                        try:
                            parsed = normalize_transform_specs(_parse_oly_line(text))
                            _add_history_entry(line)
                            return parsed
                        except Exception:
                            pass

                lines.append(line)
                prompt = continuation
