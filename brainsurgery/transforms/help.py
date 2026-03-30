from dataclasses import dataclass
from io import StringIO
from typing import Any

from ..core import (
    StateDictProvider,
    TransformControl,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    get_transform,
    list_transforms,
    register_transform,
)
from ..engine import emit_line, emit_verbose_event
from ..expressions import get_assert_expr_help, get_assert_expr_names


class HelpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class HelpSpec:
    command: str | None = None
    subcommand: str | None = None

    def collect_models(self) -> set[str]:
        return set()


class HelpTransform(TypedTransform[HelpSpec]):
    name = "help"
    error_type = HelpTransformError
    spec_type = HelpSpec
    help_text = (
        "Shows help for commands and assert expressions.\n"
        "\n"
        "Run without arguments to list all commands. You can request help for a specific "
        "command or, for 'assert', for an individual expression operator.\n"
        "\n"
        "Examples:\n"
        "  YAML: help\n"
        "  YAML: help: copy\n"
        "  YAML: help: assert\n"
        "  YAML: help: { assert: equal }\n"
        "  OLY:  help: copy\n"
        "  OLY:  help: assert: equal"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed={"help", "assert"},
        )

    def compile(self, payload: Any, default_model: str | None) -> HelpSpec:
        del default_model

        if payload is None:
            return HelpSpec(command=None, subcommand=None)

        if isinstance(payload, str):
            cmd = payload.strip()
            if not cmd:
                raise HelpTransformError("help payload must be a non-empty string")
            return HelpSpec(command=cmd, subcommand=None)

        if isinstance(payload, dict):
            if not payload:
                return HelpSpec(command=None, subcommand=None)

            if len(payload) != 1:
                raise HelpTransformError("help mapping payload must have exactly one key")

            command, subpayload = next(iter(payload.items()))
            if not isinstance(command, str) or not command.strip():
                raise HelpTransformError("help command must be a non-empty string")

            command = command.strip()

            if subpayload is None:
                return HelpSpec(command=command, subcommand=None)

            if not isinstance(subpayload, str) or not subpayload.strip():
                raise HelpTransformError("help subcommand must be a non-empty string when provided")

            return HelpSpec(command=command, subcommand=subpayload.strip())

        raise HelpTransformError("help payload must be empty, a string, or a single-key mapping")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        typed = self.require_spec(spec)
        if typed.command is None:
            emit_verbose_event(self.name, "commands")
        elif typed.subcommand is None:
            emit_verbose_event(self.name, typed.command)
        else:
            emit_verbose_event(self.name, f"{typed.command}.{typed.subcommand}")
        self._dispatch_help(typed)

        return TransformResult(
            name=self.name,
            count=0,
            control=TransformControl.CONTINUE,
        )

    def _dispatch_help(self, spec: HelpSpec) -> None:
        if spec.command is None:
            self._print_all_commands()
            return

        if spec.command == "assert":
            if spec.subcommand is None:
                self._print_assert_help()
            else:
                self._print_assert_expr_help(spec.subcommand)
            return

        if spec.subcommand is not None:
            raise HelpTransformError(f"command {spec.command!r} does not support subcommand help")
        self._print_command_help(spec.command)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise HelpTransformError("help does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def completion_payload_start_candidates(self, prefix_text: str) -> list[str] | None:
        candidates = [name for name in list_transforms() if name.startswith(prefix_text)]
        if not prefix_text:
            return candidates + ["{ "]
        return candidates

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor
        candidate = "assert: "
        if candidate.startswith(prefix_text):
            return [candidate]
        return []

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key == "help":
            return [name for name in list_transforms() if name.startswith(prefix_text)]
        if value_key == "assert":
            return [name for name in get_assert_expr_names() if name.startswith(prefix_text)]
        return None

    def completion_committed_next_candidates(self, value_key: str | None) -> list[str] | None:
        if value_key in {"help", "assert"}:
            return ["}"]
        return None

    def _print_all_commands(self) -> None:
        lines: list[str] = []
        lines.append("Available commands:")
        for name in list_transforms():
            lines.append(f"  {name}")
        lines.append("")
        lines.append("For help on a specific command:")
        lines.append("  YAML: help: <command>")
        lines.append("  OLY:  help: <command>")
        lines.append("For help on a specific assert expression:")
        lines.append("  YAML: help: { assert: <expr> }")
        lines.append("  OLY:  help: assert: <expr>")
        self._emit_help_panel("Help for commands", lines)

    def _print_command_help(self, command_name: str) -> None:
        try:
            transform = get_transform(command_name)
        except TransformError as exc:
            raise HelpTransformError(f"unknown command: {command_name}") from exc

        schema = transform.payload_schema()
        default_mode = str(schema.default_mode)
        allowed_keys = schema.allowed_keys_for_mode(default_mode)
        required_keys = schema.required_keys_for_mode(default_mode)
        help_text = getattr(transform, "help_text", None)

        lines: list[str] = [f"Command: {command_name}"]

        if help_text:
            lines.append(help_text)
        lines.append("")
        lines.append("Syntax:")
        lines.append("  YAML: <command>: { key: value, ... }")
        lines.append("  OLY:  <command>: key: value, ...")

        lines.extend(
            self._build_key_metadata_lines(
                required_keys=required_keys,
                allowed_keys=allowed_keys,
            )
        )
        if schema.mode_key is not None:
            lines.append("")
            lines.append(f"Mode key: {schema.mode_key} (default: {default_mode})")
            for mode_name in sorted(schema.modes()):
                lines.append(f"  Mode '{mode_name}':")
                mode_required = sorted(schema.required_keys_for_mode(mode_name))
                mode_allowed = sorted(schema.allowed_keys_for_mode(mode_name))
                lines.append(
                    f"    required={mode_required if mode_required else []}, "
                    f"allowed={mode_allowed if mode_allowed else []}"
                )
        self._emit_help_panel(f"Help for {command_name}", lines)

    def _print_assert_help(self) -> None:
        try:
            transform = get_transform("assert")
        except TransformError:
            transform = None

        lines: list[str] = ["Command: assert"]
        if transform is not None:
            help_text = getattr(transform, "help_text", None)
            if help_text:
                lines.append(help_text)
        lines.append("")
        lines.append("Syntax:")
        lines.append("  YAML: assert: { <expr>: { ... } }")
        lines.append("  OLY:  assert: <expr>: { ... }")
        lines.append("")
        lines.append("Supported assert expression operators:")
        for name in get_assert_expr_names():
            meta = get_assert_expr_help(name)
            if isinstance(meta, dict):
                continue
            if meta.description:
                lines.append(f"  {name}: {meta.description}")
            else:
                lines.append(f"  {name}")

        lines.append("")
        lines.append("For help on a specific assert expression:")
        lines.append("  YAML: help: { assert: <expr> }")
        lines.append("  OLY:  help: assert: <expr>")
        self._emit_help_panel("Help for assert", lines)

    def _print_assert_expr_help(self, expr_name: str) -> None:
        meta = get_assert_expr_help(expr_name)
        if isinstance(meta, dict):
            raise TransformError(f"unknown assert op: {expr_name!r}")

        lines: list[str] = [
            f"Assert expression: {meta.name}",
            f"Payload: {meta.payload_kind}",
        ]
        if meta.description:
            lines.append(meta.description)

        lines.extend(
            self._build_key_metadata_lines(
                required_keys=meta.required_keys,
                allowed_keys=meta.allowed_keys,
            )
        )
        self._emit_help_panel(f"Help for assert.{meta.name}", lines)

    def _build_key_metadata_lines(
        self,
        *,
        required_keys: set[str] | None,
        allowed_keys: set[str] | None,
    ) -> list[str]:
        if allowed_keys is None and required_keys is None:
            return ["Key metadata: unavailable"]

        lines: list[str] = []
        required = sorted(required_keys or set())
        allowed = sorted(allowed_keys or set())
        optional = [key for key in allowed if key not in required]

        if required_keys is not None:
            if required:
                lines.append("Required keys:")
                for key in required:
                    lines.append(f"  - {key}")
            else:
                lines.append("Required keys: none")

        if allowed_keys is not None:
            if optional:
                lines.append("Optional keys:")
                for key in optional:
                    lines.append(f"  - {key}")
            else:
                lines.append("Optional keys: none")

            if allowed:
                lines.append("All allowed keys:")
                for key in allowed:
                    lines.append(f"  - {key}")
            else:
                lines.append("All allowed keys: none")

        return lines

    def _emit_help_panel(self, title: str, lines: list[str]) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel
        except Exception:
            emit_line(title)
            for line in lines:
                emit_line(line)
            return

        buffer = StringIO()
        console = Console(
            file=buffer,
            color_system=None,
            force_terminal=False,
            width=100,
        )
        console.print(Panel.fit("\n".join(lines), title=title, border_style="cyan"))
        for line in buffer.getvalue().splitlines():
            emit_line(line)

    def _emit_key_metadata(
        self,
        *,
        required_keys: set[str] | None,
        allowed_keys: set[str] | None,
    ) -> None:
        for line in self._build_key_metadata_lines(
            required_keys=required_keys,
            allowed_keys=allowed_keys,
        ):
            emit_line(line)


register_transform(HelpTransform())
