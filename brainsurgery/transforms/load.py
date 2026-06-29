from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..core import (
    StateDictLike,
    StateDictProvider,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    complete_filesystem_paths,
    ensure_mapping_payload,
    parse_model_expr,
    register_transform,
    require_nonempty_string,
    validate_payload_schema,
)
from ..engine import (
    BaseStateDictProvider,
    ProviderError,
    emit_verbose_event,
    get_model_runtime_metadata,
    get_or_create_alias_state_dict,
    get_runtime_flags,
    load_tensor_from_path,
    set_model_runtime_metadata,
)


class LoadTransformError(TransformError):
    pass


LoadFormat = Literal["auto", "torch", "safetensors", "numpy"]


@dataclass(frozen=True)
class LoadSpec:
    path: Path
    alias: str
    tensor_name: str | None
    format: LoadFormat
    backend: str = "auto"
    mode: str = "checkpoint"
    weights: Path | None = None

    def collect_models(self) -> set[str]:
        # load can introduce a new model alias
        return set()


class LoadTransform(TypedTransform[LoadSpec]):
    name = "load"
    error_type = LoadTransformError
    spec_type = LoadSpec
    help_text = (
        "Loads either a full state_dict or a single tensor from disk.\n"
        "\n"
        "With backend=auto (default), mode is inferred from path + to:\n"
        "  - to present -> tensor load\n"
        "  - otherwise -> checkpoint/state_dict load\n"
        "\n"
        "Without 'to', loads a full state_dict into 'alias'. With 'to', loads one tensor "
        "into a tensor name (optionally with alias in 'to', e.g. model::name).\n"
        "\n"
        "Examples:\n"
        "  load: { path: /tmp/a.safetensors, alias: a }\n"
        "  load: { path: /tmp/tensor.npy, to: model::embed.weight }\n"
        "  load: /tmp/model.safetensors"
    )

    def completion_reference_keys(self) -> list[str]:
        return ["to"]

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key="backend",
            default_mode="auto",
            common_required={"path"},
            common_allowed={"path", "alias", "backend"},
            mode_required={},
            mode_allowed_extra={
                "auto": {"to", "format"},
                "checkpoint": {"format"},
                "tensor": {"to", "format"},
            },
        )

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        if value_key in {"path", "weights"}:
            return complete_filesystem_paths(prefix_text)
        if value_key == "alias":
            return [alias for alias in model_aliases if alias.startswith(prefix_text)]
        if value_key == "format":
            return [
                name
                for name in ("auto", "torch", "safetensors", "numpy")
                if name.startswith(prefix_text)
            ]
        if value_key == "backend":
            return [
                name
                for name in ("auto", "checkpoint", "tensor")
                if name.startswith(prefix_text)
            ]
        return None

    def compile(self, payload: Any, default_model: str | None) -> LoadSpec:
        if isinstance(payload, str):
            payload = {"path": payload}
        payload = ensure_mapping_payload(payload, self.name)
        backend = validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=LoadTransformError,
        )

        path = Path(require_nonempty_string(payload, op_name=self.name, key="path"))

        raw_alias = payload.get("alias")
        if raw_alias is not None and (not isinstance(raw_alias, str) or not raw_alias):
            raise LoadTransformError("load.alias must be a non-empty string when provided")

        raw_format = payload.get("format", "auto")
        if not isinstance(raw_format, str) or not raw_format:
            raise LoadTransformError("load.format must be a non-empty string when provided")
        fmt_raw = raw_format.strip().lower()
        if fmt_raw not in {"auto", "torch", "safetensors", "numpy"}:
            raise LoadTransformError("load.format must be one of: auto, torch, safetensors, numpy")
        fmt = cast(LoadFormat, fmt_raw)

        raw_weights = payload.get("weights")
        if raw_weights is not None and (not isinstance(raw_weights, str) or not raw_weights):
            raise LoadTransformError("load.weights must be a non-empty string when provided")
        weights = Path(raw_weights) if isinstance(raw_weights, str) else None

        alias_default = raw_alias or default_model or "model"
        tensor_name: str | None = None
        raw_to = payload.get("to")
        if raw_to is not None:
            if not isinstance(raw_to, str) or not raw_to:
                raise LoadTransformError("load.to must be a non-empty string when provided")
            target_ref = parse_model_expr(raw_to, default_model=alias_default)
            if target_ref.slice_spec is not None:
                raise LoadTransformError("load.to must not be sliced")
            if not isinstance(target_ref.expr, str):
                raise LoadTransformError("load.to must resolve to a single tensor name")
            if raw_alias is not None and target_ref.model != raw_alias:
                raise LoadTransformError("load.alias conflicts with model alias in load.to")
            assert target_ref.model is not None
            alias_default = target_ref.model
            tensor_name = target_ref.expr

        mode = self._resolve_mode(
            backend=backend,
            path=path,
            has_to=(tensor_name is not None),
        )

        if mode == "tensor":
            if weights is not None:
                raise LoadTransformError("load.weights is not supported for tensor load mode")
        elif mode == "checkpoint":
            if weights is not None:
                raise LoadTransformError("load.weights is no longer supported")
            if fmt != "auto":
                raise LoadTransformError(
                    "load.format is only supported for tensor loads (with load.to)"
                )
        else:  # pragma: no cover
            raise LoadTransformError(f"Unsupported load mode: {mode}")

        return LoadSpec(
            path=path,
            alias=alias_default,
            tensor_name=tensor_name,
            format=fmt,
            backend=backend,
            mode=mode,
            weights=weights,
        )

    def _resolve_mode(self, *, backend: str, path: Path, has_to: bool) -> str:
        if backend == "tensor":
            if not has_to:
                raise LoadTransformError("load.backend=tensor requires load.to")
            return "tensor"
        if backend == "checkpoint":
            if has_to:
                raise LoadTransformError("load.backend=checkpoint does not support load.to")
            return "checkpoint"
        if backend == "axon":
            raise LoadTransformError("load.backend=axon was removed with the legacy Synapse runtime")
        # auto
        if has_to:
            return "tensor"
        if path.suffix.lower() == ".axon":
            raise LoadTransformError(".axon model loading was removed with the legacy Synapse runtime")
        return "checkpoint"

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        dry_run = get_runtime_flags().dry_run

        if typed.tensor_name is None:
            if not isinstance(provider, BaseStateDictProvider):
                raise LoadTransformError(
                    "load requires a provider that supports creating new aliases"
                )
            try:
                if dry_run:
                    loaded_state_dict = provider.load_state_dict_from_checkpoint_path(typed.path)
                else:
                    loaded_state_dict = provider.load_alias_from_path(typed.alias, typed.path)
            except ProviderError as exc:
                message = str(exc).replace("model alias", "load alias")
                raise LoadTransformError(message) from exc
            except RuntimeError as exc:
                raise LoadTransformError(str(exc)) from exc
            if not dry_run:
                self._set_runtime_metadata_for_checkpoint_load(
                    provider=provider,
                    alias=typed.alias,
                    path=typed.path,
                )
            emit_verbose_event(self.name, f"{typed.path} -> alias {typed.alias}")
            return TransformResult(name=self.name, count=len(loaded_state_dict))

        try:
            tensor = load_tensor_from_path(typed.path, format=typed.format)
        except RuntimeError as exc:
            raise LoadTransformError(str(exc)) from exc
        if (
            dry_run
            and isinstance(provider, BaseStateDictProvider)
            and not provider.has_model_alias(typed.alias)
        ):
            state_dict: StateDictLike = cast(StateDictLike, {})
        else:
            state_dict = get_or_create_alias_state_dict(
                provider,
                typed.alias,
                error_type=LoadTransformError,
                op_name=self.name,
            )
        if typed.tensor_name in state_dict:
            raise LoadTransformError(
                f"load destination already exists: {typed.alias}::{typed.tensor_name}"
            )
        if not dry_run:
            state_dict[typed.tensor_name] = tensor
        emit_verbose_event(self.name, f"{typed.path} -> {typed.alias}::{typed.tensor_name}")
        return TransformResult(name=self.name, count=1)

    def _set_runtime_metadata_for_checkpoint_load(
        self,
        *,
        provider: StateDictProvider,
        alias: str,
        path: Path,
    ) -> None:
        existing = get_model_runtime_metadata(provider, alias)
        if isinstance(existing, dict):
            runtime = existing.get("runtime")
            program = existing.get("program")
            if runtime == "hf" and isinstance(program, str) and program:
                return
        hf_program = _infer_hf_program_path(path)
        if hf_program is not None:
            set_model_runtime_metadata(
                provider,
                alias,
                {"runtime": "hf", "program": str(hf_program)},
            )

    def _infer_output_model(self, spec: object) -> str:
        return self.require_spec(spec).alias


register_transform(LoadTransform())


def _infer_hf_program_path(path: Path) -> Path | None:
    if path.is_dir():
        return path
    if path.is_file():
        candidate = path.parent
        if (candidate / "config.json").is_file():
            return candidate
    return None
