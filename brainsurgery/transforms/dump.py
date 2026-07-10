import json
from dataclasses import dataclass
from typing import Any

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    UnaryTransform,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    register_transform,
    state_dict_for_ref,
    unary_view_for_ref_name,
    validate_payload_schema,
)
from ..engine import emit_line, emit_verbose_unary_activity, render_tree, summarize_tensor, tqdm


class DumpTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DumpSpec:
    target_ref: TensorRef | None
    format: str
    verbosity: str
    dump_all_models: bool = False
    default_model_hint: str | None = None

    def collect_models(self) -> set[str]:
        if self.dump_all_models:
            return {self.default_model_hint} if self.default_model_hint is not None else set()
        if self.target_ref is None:
            return set()
        return {must_model(self.target_ref)}


class DumpTransform(UnaryTransform[DumpSpec]):  # type: ignore[type-var]
    name = "dump"
    error_type = DumpTransformError
    spec_type = DumpSpec
    allowed_keys = {"target", "format", "verbosity"}
    required_keys = set()
    slice_policy = "allow"
    progress_desc = "Dumping tensors"
    help_text = (
        "Displays tensors selected by 'target' without modifying them.\n"
        "\n"
        "When 'target' is omitted, dumps all tensors across all loaded model aliases.\n"
        "\n"
        "Targets may be specified by name or pattern. Slices are written after '::', "
        "for example 'ln_f.weight::[:8]'. 'format' controls layout: 'tree' (default), "
        "'compact', or 'json'. 'verbosity' controls content: 'shape', 'stat' (default), "
        "or 'full'.\n"
        "\n"
        "Examples:\n"
        "  dump: { target: ln_f.weight }\n"
        "  dump: { target: '.*weight', format: compact, verbosity: shape }\n"
        "  dump: { target: 'h.0.attn.c_attn.weight::[:, :10]', format: json, verbosity: stat }\n"
        "  dump: { target: 'ln_f.weight::[:8]', format: tree, verbosity: full }"
    )

    def compile(self, payload: dict, default_model: str | None) -> DumpSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        target_ref: TensorRef | None = None
        dump_all_models = "target" not in payload
        if not dump_all_models:
            raw_target = self.require_target_expr(payload)
            target_ref = parse_model_expr(raw_target, default_model=default_model)
            self.validate_target_ref(target_ref)
            assert target_ref.model is not None

        return self.build_spec(
            target_ref=target_ref,
            payload=payload,
            dump_all_models=dump_all_models,
            default_model_hint=default_model,
        )

    def build_spec(
        self,
        target_ref: TensorRef | None,
        payload: dict,
        *,
        dump_all_models: bool = False,
        default_model_hint: str | None = None,
    ) -> DumpSpec:
        raw_format = payload.get("format", "compact")
        if not isinstance(raw_format, str) or not raw_format:
            raise DumpTransformError("dump.format must be a non-empty string")

        fmt = raw_format.strip().lower()
        if fmt not in {"json", "tree", "compact"}:
            raise DumpTransformError("dump.format must be one of: json, tree, compact")

        raw_verbosity = payload.get("verbosity", "shape")
        if not isinstance(raw_verbosity, str) or not raw_verbosity:
            raise DumpTransformError("dump.verbosity must be a non-empty string")

        verbosity = raw_verbosity.strip().lower()
        if verbosity not in {"shape", "stat", "full"}:
            raise DumpTransformError("dump.verbosity must be one of: shape, stat, full")

        return DumpSpec(
            target_ref=target_ref,
            format=fmt,
            verbosity=verbosity,
            dump_all_models=dump_all_models,
            default_model_hint=default_model_hint,
        )

    def apply_to_target(self, spec: DumpSpec, name: str, provider: StateDictProvider) -> None:
        raise AssertionError("DumpTransform overrides apply() and does not use apply_to_target()")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise DumpTransformError("dump does not infer an output model")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        tree: dict[str, Any] = {}
        trees_by_alias: dict[str, dict[str, Any]] = {}
        total = 0

        if typed.dump_all_models:
            aliases = _resolve_model_aliases(provider, typed.default_model_hint)
            for alias in aliases:
                sd = provider.get_state_dict(alias)
                alias_tree: dict[str, Any] = {}
                for name in tqdm(sorted(sd.keys()), desc=self.progress_desc, unit="tensor"):
                    tensor = sd[name]
                    access_counts = _maybe_get_access_counts(sd, name, verbosity=typed.verbosity)
                    summary = summarize_tensor(
                        tensor, verbosity=typed.verbosity, access_counts=access_counts
                    )
                    insert_into_tree(
                        tree,
                        [alias, *name.split(".")],
                        summary,
                    )
                    insert_into_tree(alias_tree, name.split("."), summary)
                    emit_verbose_unary_activity(self.name, f"{alias}::{name}")
                    total += 1
                trees_by_alias[alias] = alias_tree
        else:
            if typed.target_ref is None:
                raise DumpTransformError("dump target missing")
            sd = state_dict_for_ref(provider, typed.target_ref)
            targets = self.resolve_targets(typed, provider)

            for name in tqdm(targets, desc=self.progress_desc, unit="tensor"):
                _sd, view = unary_view_for_ref_name(provider, typed.target_ref, name)
                access_counts = _maybe_get_access_counts(sd, name, verbosity=typed.verbosity)
                insert_into_tree(
                    tree,
                    name.split("."),
                    summarize_tensor(view, verbosity=typed.verbosity, access_counts=access_counts),
                )
                emit_verbose_unary_activity(self.name, name)
                total += 1

        if typed.format == "json":
            if typed.dump_all_models:
                if not trees_by_alias:
                    emit_line("{}")
                else:
                    for alias in sorted(trees_by_alias):
                        emit_line(
                            json.dumps(
                                {alias: trees_by_alias[alias]},
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                        )
            else:
                emit_line(json.dumps(tree, separators=(",", ":"), sort_keys=True))
        elif typed.format == "tree":
            if typed.dump_all_models:
                blocks = [
                    render_tree({alias: trees_by_alias[alias]}, compact=False)
                    for alias in sorted(trees_by_alias)
                ]
                emit_line("\n\n".join(blocks))
            else:
                emit_line(render_tree(tree, compact=False))
        else:
            if typed.dump_all_models:
                blocks = [
                    render_tree({alias: trees_by_alias[alias]}, compact=True)
                    for alias in sorted(trees_by_alias)
                ]
                emit_line("\n\n".join(blocks))
            else:
                emit_line(render_tree(tree, compact=True))

        return TransformResult(name=self.name, count=total)


def _resolve_model_aliases(
    provider: StateDictProvider, default_model_hint: str | None
) -> list[str]:
    list_aliases = getattr(provider, "list_model_aliases", None)
    if callable(list_aliases):
        aliases = sorted(alias for alias in list_aliases() if isinstance(alias, str) and alias)
        return aliases
    if default_model_hint is not None:
        return [default_model_hint]
    return []


def insert_into_tree(tree: dict[str, Any], parts: list[str], leaf: Any) -> None:
    node: Any = tree

    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        next_is_index = i + 1 < len(parts) and parts[i + 1].isdigit()

        if part.isdigit():
            idx = int(part)

            if not isinstance(node, list):
                raise DumpTransformError("invalid tree structure while building dump")

            while len(node) <= idx:
                node.append(None)

            if is_last:
                node[idx] = leaf
                return

            child = node[idx]
            if child is None:
                child = [] if next_is_index else {}
                node[idx] = child
            elif not isinstance(child, dict | list):
                raise DumpTransformError("invalid tree structure while building dump")

            node = child
            continue

        if not isinstance(node, dict):
            raise DumpTransformError("invalid tree structure while building dump")

        if is_last:
            node[part] = leaf
            return

        child = node.get(part)
        if child is None:
            child = [] if next_is_index else {}
            node[part] = child
        elif not isinstance(child, dict | list):
            raise DumpTransformError("invalid tree structure while building dump")

        node = child


def _maybe_get_access_counts(
    state_dict: Any,
    key: str,
    *,
    verbosity: str,
) -> dict[str, int] | None:
    if verbosity not in {"stat", "full"}:
        return None
    access_counts = getattr(state_dict, "access_counts", None)
    if not callable(access_counts):
        return None
    return access_counts(key)


register_transform(DumpTransform())
