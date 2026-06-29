from collections.abc import Mapping

from ..core import StateDictLike, StateDictProvider, TransformError


def _is_base_provider_instance(provider: object) -> bool:
    try:
        from .providers import BaseStateDictProvider
    except Exception:
        return False
    return isinstance(provider, BaseStateDictProvider)


def iter_alias_mappings(provider: StateDictProvider) -> list[tuple[str, dict[str, object]]]:
    mappings: list[tuple[str, dict[str, object]]] = []
    for attr_name in ("model_paths", "state_dicts", "_state_dicts"):
        value = getattr(provider, attr_name, None)
        if isinstance(value, dict):
            mappings.append((attr_name, value))
    return mappings


def list_model_aliases(provider: StateDictProvider | None) -> set[str]:
    if provider is None:
        return set()

    if _is_base_provider_instance(provider):
        list_aliases = getattr(provider, "list_model_aliases", None)
        if callable(list_aliases):
            return {str(alias) for alias in list_aliases()}
        return set()

    list_aliases = getattr(provider, "list_model_aliases", None)
    if callable(list_aliases):
        loaded_aliases = list_aliases()
        if isinstance(loaded_aliases, set):
            return {str(alias) for alias in loaded_aliases}
        return {str(alias) for alias in loaded_aliases}

    aliases: set[str] = set()
    for _, mapping in iter_alias_mappings(provider):
        aliases.update(str(alias) for alias in mapping.keys())
    return aliases


def set_model_runtime_metadata(
    provider: StateDictProvider,
    alias: str,
    metadata: dict[str, object],
) -> None:
    setter = getattr(provider, "set_model_runtime_metadata", None)
    if callable(setter):
        setter(alias, dict(metadata))
        return
    bag = getattr(provider, "model_runtime_metadata", None)
    if isinstance(bag, dict):
        bag[alias] = dict(metadata)
        return
    setattr(provider, "model_runtime_metadata", {alias: dict(metadata)})


def get_model_runtime_metadata(
    provider: StateDictProvider,
    alias: str,
) -> dict[str, object] | None:
    getter = getattr(provider, "get_model_runtime_metadata", None)
    if callable(getter):
        result = getter(alias)
        return dict(result) if isinstance(result, dict) else None
    bag = getattr(provider, "model_runtime_metadata", None)
    if not isinstance(bag, dict):
        return None
    value = bag.get(alias)
    return dict(value) if isinstance(value, dict) else None


def _has_model_alias(provider: StateDictProvider, alias: str) -> bool:
    if _is_base_provider_instance(provider):
        has_model_alias = getattr(provider, "has_model_alias", None)
        if callable(has_model_alias):
            return bool(has_model_alias(alias))
        return alias in list_model_aliases(provider)
    return alias in list_model_aliases(provider)


def get_or_create_alias_state_dict(
    provider: StateDictProvider,
    alias: str,
    *,
    error_type: type[TransformError],
    op_name: str,
) -> StateDictLike:
    if _is_base_provider_instance(provider):
        get_or_create = getattr(provider, "get_or_create_alias_state_dict", None)
        if callable(get_or_create):
            return get_or_create(alias)
        return provider.get_state_dict(alias)
    if _has_model_alias(provider, alias):
        return provider.get_state_dict(alias)
    raise error_type(f"{op_name} requires a provider that supports creating new aliases")


def list_loaded_tensor_names(provider: StateDictProvider | None) -> dict[str, set[str]]:
    if provider is None:
        return {}

    if _is_base_provider_instance(provider):
        raw_state_dicts = getattr(provider, "state_dicts", None)
        if not isinstance(raw_state_dicts, Mapping):
            return {}
        items: Mapping[str, StateDictLike] = raw_state_dicts
    else:
        state_dicts = getattr(provider, "state_dicts", None)
        if not isinstance(state_dicts, dict):
            return {}
        items = state_dicts

    loaded: dict[str, set[str]] = {}
    for alias, state_dict in items.items():
        keys = getattr(state_dict, "keys", None)
        if not callable(keys):
            continue
        try:
            loaded[str(alias)] = {str(name) for name in keys()}
        except Exception:
            continue
    return loaded


def resolve_single_model_alias(
    provider: StateDictProvider,
    *,
    error_type: type[TransformError],
    op_name: str,
) -> str:
    aliases = list_model_aliases(provider)
    if len(aliases) != 1:
        raise error_type(f"{op_name}.alias is required when more than one model alias is available")
    return next(iter(aliases))


def find_alias_mapping(
    provider: StateDictProvider,
    alias: str,
    *,
    error_type: type[TransformError],
) -> tuple[str, dict[str, object], object]:
    for attr_name, mapping in iter_alias_mappings(provider):
        if alias in mapping:
            return attr_name, mapping, mapping[alias]
    raise error_type(f"unknown model prefix: {alias!r}")


def new_empty_state_dict(mappings: list[tuple[str, dict[str, object]]]) -> object:
    for _, mapping in mappings:
        for value in mapping.values():
            state_dict_type = type(value)
            try:
                return state_dict_type()
            except Exception:
                continue
    return {}


__all__ = [
    "iter_alias_mappings",
    "list_model_aliases",
    "set_model_runtime_metadata",
    "get_model_runtime_metadata",
    "get_or_create_alias_state_dict",
    "resolve_single_model_alias",
    "find_alias_mapping",
    "new_empty_state_dict",
]
