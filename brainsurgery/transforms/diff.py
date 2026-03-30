import re
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    ensure_mapping_payload,
    format_tensor_ref,
    match_expr_names,
    must_model,
    parse_model_expr,
    parse_slice,
    register_transform,
    require_nonempty_string,
    state_dict_for_ref,
    validate_payload_schema,
    view_for_ref_name,
)
from ..engine import emit_line, emit_verbose_event


class DiffTransformError(TransformError):
    pass


DiffMode = Literal["refs", "aliases"]


@dataclass(frozen=True)
class DiffSpec:
    left_ref: TensorRef
    right_ref: TensorRef
    eps: float | None
    mode: DiffMode

    def collect_models(self) -> set[str]:
        return {must_model(self.left_ref), must_model(self.right_ref)}


class DiffTransform(TypedTransform[DiffSpec]):
    name = "diff"
    error_type = DiffTransformError
    spec_type = DiffSpec
    allowed_keys = {"mode", "left", "right", "left_alias", "right_alias", "eps"}
    help_text = (
        "Compares two tensor sets and reports missing names on both sides plus tensors "
        "whose contents differ.\n"
        "\n"
        "Modes:\n"
        "  - refs (default): compare tensors selected by 'left' and 'right'\n"
        "  - aliases: compare all tensors from 'left_alias' and 'right_alias'\n"
        "\n"
        "Comparison is symmetric by tensor name. Shared names are compared by shape, dtype, "
        "device, then value equality. When 'eps' is provided, value comparisons use absolute "
        "tolerance.\n"
        "\n"
        "Examples:\n"
        "  diff: { left: base::'.*', right: edited::'.*' }\n"
        "  diff: { left: base::ln_f\\..*, right: edited::ln_f\\..*, eps: 1e-6 }\n"
        "  diff: { mode: aliases, left_alias: base, right_alias: edited }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key="mode",
            default_mode="refs",
            common_required=set(),
            common_allowed={"mode", "eps"},
            mode_required={
                "refs": {"left", "right"},
                "aliases": {"left_alias", "right_alias"},
            },
            mode_allowed_extra={},
        )

    def compile(self, payload: Any, default_model: str | None) -> DiffSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        raw_mode = payload.get("mode", "refs")
        if not isinstance(raw_mode, str) or not raw_mode:
            raise DiffTransformError("diff.mode must be a non-empty string when provided")
        mode = raw_mode.strip().lower()
        if mode not in {"refs", "aliases"}:
            raise DiffTransformError("diff.mode must be one of: refs, aliases")

        eps = _compile_eps(payload.get("eps"))

        if mode == "refs":
            left_ref = parse_model_expr(payload["left"], default_model=default_model)
            right_ref = parse_model_expr(payload["right"], default_model=default_model)
            _validate_slice(left_ref)
            _validate_slice(right_ref)
            return DiffSpec(left_ref=left_ref, right_ref=right_ref, eps=eps, mode="refs")

        left_alias = require_nonempty_string(payload, op_name=self.name, key="left_alias")
        right_alias = require_nonempty_string(payload, op_name=self.name, key="right_alias")
        return DiffSpec(
            left_ref=TensorRef(model=left_alias, expr=".*"),
            right_ref=TensorRef(model=right_alias, expr=".*"),
            eps=eps,
            mode="aliases",
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        left_names = set(_resolve_names(typed.left_ref, provider))
        right_names = set(_resolve_names(typed.right_ref, provider))

        missing_on_left = sorted(right_names - left_names)
        missing_on_right = sorted(left_names - right_names)
        shared = sorted(left_names & right_names)
        differing = _collect_differences(shared, typed, provider)

        emit_line(
            f"Diff: {format_tensor_ref(typed.left_ref)} <-> {format_tensor_ref(typed.right_ref)}"
        )
        _echo_names("Missing on left", missing_on_left)
        _echo_names("Missing on right", missing_on_right)
        _echo_differences(differing)

        findings = len(missing_on_left) + len(missing_on_right) + len(differing)
        if findings == 0:
            emit_line("No differences found.")
        emit_verbose_event(self.name, f"{findings} finding(s)")

        return TransformResult(name=self.name, count=findings)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise DiffTransformError("diff does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def completion_reference_keys(self) -> list[str]:
        return ["left", "right"]

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        used_keys = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", before_cursor))
        mode = _diff_mode(before_cursor)
        if mode == "aliases":
            options = ["mode: ", "left_alias: ", "right_alias: ", "eps: "]
        else:
            options = ["mode: ", "left: ", "right: ", "eps: "]
        filtered = [
            candidate
            for candidate in options
            if candidate[:-2] not in used_keys and candidate.startswith(prefix_text)
        ]
        if filtered:
            return filtered
        if options and not all(option[:-2] in used_keys for option in options):
            return []
        return ["}"]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        if value_key == "mode":
            return [mode for mode in ("refs", "aliases") if mode.startswith(prefix_text)]
        if value_key in {"left_alias", "right_alias"}:
            return [alias for alias in model_aliases if alias.startswith(prefix_text)]
        return None


def _compile_eps(raw_eps: Any) -> float | None:
    if raw_eps is None:
        return None
    if isinstance(raw_eps, bool) or not isinstance(raw_eps, int | float):
        raise DiffTransformError("diff.eps must be a non-negative number")
    eps = float(raw_eps)
    if eps < 0:
        raise DiffTransformError("diff.eps must be a non-negative number")
    return eps


def _validate_slice(ref: TensorRef) -> None:
    if ref.slice_spec is not None:
        parse_slice(ref.slice_spec)


def _resolve_names(ref: TensorRef, provider: StateDictProvider) -> list[str]:
    state_dict = state_dict_for_ref(provider, ref)
    try:
        return match_expr_names(
            expr=ref.expr,
            names=state_dict.keys(),
            op_name="diff",
            role="target",
        )
    except TransformError as exc:
        raise DiffTransformError(str(exc)) from exc


def _collect_differences(
    shared: list[str],
    spec: DiffSpec,
    provider: StateDictProvider,
) -> list[tuple[str, str]]:
    differing: list[tuple[str, str]] = []
    for name in shared:
        _left_sd, left = view_for_ref_name(provider, spec.left_ref, name)
        _right_sd, right = view_for_ref_name(provider, spec.right_ref, name)
        reason = _difference_reason(left, right, eps=spec.eps)
        if reason is not None:
            differing.append((name, reason))
    return differing


def _difference_reason(left: torch.Tensor, right: torch.Tensor, *, eps: float | None) -> str | None:
    if left.shape != right.shape:
        return f"shape {tuple(left.shape)} != {tuple(right.shape)}"
    if left.dtype != right.dtype:
        return f"dtype {left.dtype} != {right.dtype}"
    if left.device != right.device:
        return f"device {left.device} != {right.device}"

    if eps is None:
        if torch.equal(left, right):
            return None
        return "values differ"

    if left.is_complex():
        diff = torch.abs(left.to(torch.complex128) - right.to(torch.complex128))
    else:
        diff = torch.abs(left.to(torch.float64) - right.to(torch.float64))

    if bool(torch.all(diff <= eps).item()):
        return None

    max_abs_diff = float(diff.max().item())
    return f"values differ (max_abs_diff={max_abs_diff:g}, eps={eps:g})"


def _echo_names(title: str, names: list[str]) -> None:
    emit_line(f"{title}:")
    if not names:
        emit_line("  (none)")
        return
    for name in names:
        emit_line(f"  - {name}")


def _echo_differences(differing: list[tuple[str, str]]) -> None:
    emit_line("Differing:")
    if not differing:
        emit_line("  (none)")
        return
    for name, reason in differing:
        emit_line(f"  - {name}: {reason}")


def _diff_mode(before_cursor: str) -> str | None:
    match = re.search(r"\bmode\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", before_cursor)
    if match is None:
        return None
    return match.group(1).lower()


register_transform(DiffTransform())
