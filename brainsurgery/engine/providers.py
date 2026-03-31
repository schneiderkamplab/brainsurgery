import logging
from pathlib import Path

import brainsurgery.engine.state_dicts as _state_dicts

from ..core import StateDictLike
from .arena import ProviderError, _SegmentedFileBackedArena
from .checkpoint_io import _load_state_dict_from_path, persist_state_dict
from .output_model import _infer_output_model
from .output_paths import _resolve_output_destination, parse_shard_size
from .plan import SurgeryPlan
from .state_dicts import _ArenaStateDict, _InMemoryStateDict

logger = logging.getLogger("brainsurgery")


class BaseStateDictProvider:
    def __init__(self, model_paths: dict[str, Path], max_io_workers: int):
        self.model_paths = model_paths
        self.max_io_workers = max_io_workers
        self.state_dicts: dict[str, StateDictLike] = {}
        self.model_runtime_metadata: dict[str, dict[str, object]] = {}

    def get_state_dict(self, model: str) -> StateDictLike:
        raise NotImplementedError

    def create_state_dict(self) -> StateDictLike:
        raise NotImplementedError

    def list_model_aliases(self) -> set[str]:
        return set(self.model_paths) | set(self.state_dicts)

    def has_model_alias(self, model: str) -> bool:
        return model in self.list_model_aliases()

    def attach_state_dict(self, model: str, state_dict: StateDictLike) -> None:
        self.state_dicts[model] = state_dict
        self.model_paths.pop(model, None)

    def load_state_dict_from_checkpoint_path(self, path: Path) -> StateDictLike:
        state_dict = self.create_state_dict()
        _load_state_dict_from_path(path, state_dict, max_io_workers=self.max_io_workers)
        return state_dict

    def load_alias_from_path(self, model: str, path: Path) -> StateDictLike:
        if self.has_model_alias(model):
            raise ProviderError(f"model alias already exists: {model!r}")
        state_dict = self.load_state_dict_from_checkpoint_path(path)
        self.attach_state_dict(model, state_dict)
        return state_dict

    def get_or_create_alias_state_dict(self, model: str) -> StateDictLike:
        if self.has_model_alias(model):
            return self.get_state_dict(model)
        state_dict = self.create_state_dict()
        self.attach_state_dict(model, state_dict)
        return state_dict

    def set_model_runtime_metadata(self, model: str, metadata: dict[str, object]) -> None:
        self.model_runtime_metadata[model] = dict(metadata)

    def get_model_runtime_metadata(self, model: str) -> dict[str, object] | None:
        value = self.model_runtime_metadata.get(model)
        return dict(value) if isinstance(value, dict) else None

    def _get_or_load_state_dict(
        self,
        model: str,
        *,
        loaded_log_message: str,
    ) -> StateDictLike:
        if model in self.state_dicts:
            return self.state_dicts[model]

        if model not in self.model_paths:
            raise ProviderError(f"unknown model alias: {model!r}")

        path = self.model_paths[model]
        logger.info("Opening cranium for brain '%s' at %s", model, path)

        sd = self.load_state_dict_from_checkpoint_path(path)
        self.state_dicts[model] = sd
        logger.info(loaded_log_message, model, len(sd))

        return sd

    def close(self) -> None:
        pass

    def save_output(
        self,
        plan: SurgeryPlan,
        *,
        default_shard_size: str,
        max_io_workers: int,
    ) -> Path:
        if plan.output is None:
            raise ProviderError("save_output requires plan.output")

        output_model = _infer_output_model(plan, self)
        state_dict = self.get_state_dict(output_model)

        output_path, output_format, shard_size = _resolve_output_destination(
            plan.output,
            default_shard_size=default_shard_size,
        )

        logger.info(
            "Closing incision and preserving brain '%s' to %s (%s)",
            output_model,
            output_path,
            output_format,
        )

        written_path = persist_state_dict(
            dict(state_dict.items()),
            output_path=output_path,
            output_format=output_format,
            shard_size=shard_size,
            sharded_output_root=plan.output.path,
            max_io_workers=max_io_workers,
        )
        if shard_size is None:
            logger.info("Patient stable. Preserved %d tensors at %s", len(state_dict), written_path)
        else:
            logger.info(
                "Patient stable. Wrote %d tensors across sharded safetensors in %s",
                len(state_dict),
                written_path,
            )
        return written_path


class InMemoryStateDictProvider(BaseStateDictProvider):
    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message="Brain '%s' exposed: %d tensors laid out on the operating table",
        )
        assert isinstance(state_dict, _InMemoryStateDict)
        return state_dict

    def create_state_dict(self) -> _InMemoryStateDict:
        return _InMemoryStateDict()


class ArenaStateDictProvider(BaseStateDictProvider):
    def __init__(
        self,
        model_paths: dict[str, Path],
        *,
        arena: _SegmentedFileBackedArena,
        max_io_workers: int,
    ):
        super().__init__(model_paths, max_io_workers=max_io_workers)
        self.arena = arena

    def close(self) -> None:
        self.arena.close()

    def get_state_dict(self, model: str) -> _ArenaStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message=(
                "Brain '%s' transferred to surgical arena: %d tensors laid out on the operating table"
            ),
        )
        assert isinstance(state_dict, _ArenaStateDict)
        return state_dict

    def create_state_dict(self) -> _ArenaStateDict:
        return _ArenaStateDict(self.arena)


class GpuCachedStateDictProvider(BaseStateDictProvider):
    def __init__(
        self,
        backing_provider: BaseStateDictProvider,
        *,
        cache_config: _state_dicts.GpuCacheConfig,
    ):
        super().__init__(
            backing_provider.model_paths,
            max_io_workers=backing_provider.max_io_workers,
        )
        self._backing_provider = backing_provider
        # keep alias maps shared with the backing provider
        self.model_paths = backing_provider.model_paths
        self.state_dicts = backing_provider.state_dicts
        self.model_runtime_metadata = backing_provider.model_runtime_metadata
        self._cache_config = cache_config
        self._wrapped_state_dicts: dict[str, _state_dicts.GpuCachedStateDict] = {}

    def get_state_dict(self, model: str) -> _state_dicts.GpuCachedStateDict:
        wrapped = self._wrapped_state_dicts.get(model)
        if wrapped is not None:
            return wrapped
        raw_state_dict = self._backing_provider.get_state_dict(model)
        wrapped = _state_dicts.GpuCachedStateDict(
            raw_state_dict,
            config=self._cache_config,
            cache_name=model,
        )
        self._wrapped_state_dicts[model] = wrapped
        return wrapped

    def create_state_dict(self) -> _state_dicts.GpuCachedStateDict:
        raw_state_dict = self._backing_provider.create_state_dict()
        return _state_dicts.GpuCachedStateDict(
            raw_state_dict,
            config=self._cache_config,
            cache_name="anonymous",
        )

    def list_model_aliases(self) -> set[str]:
        return self._backing_provider.list_model_aliases()

    def has_model_alias(self, model: str) -> bool:
        return self._backing_provider.has_model_alias(model)

    def attach_state_dict(self, model: str, state_dict: StateDictLike) -> None:
        if isinstance(state_dict, _state_dicts.GpuCachedStateDict):
            raw_state_dict = state_dict.backing_state_dict
            self._wrapped_state_dicts[model] = state_dict
        else:
            raw_state_dict = state_dict
            self._wrapped_state_dicts.pop(model, None)
        self._backing_provider.attach_state_dict(model, raw_state_dict)

    def load_state_dict_from_checkpoint_path(self, path: Path) -> _state_dicts.GpuCachedStateDict:
        raw_state_dict = self._backing_provider.load_state_dict_from_checkpoint_path(path)
        return _state_dicts.GpuCachedStateDict(
            raw_state_dict,
            config=self._cache_config,
            cache_name=path.name,
        )

    def load_alias_from_path(self, model: str, path: Path) -> _state_dicts.GpuCachedStateDict:
        raw_state_dict = self._backing_provider.load_alias_from_path(model, path)
        wrapped = _state_dicts.GpuCachedStateDict(
            raw_state_dict,
            config=self._cache_config,
            cache_name=model,
        )
        self._wrapped_state_dicts[model] = wrapped
        return wrapped

    def get_or_create_alias_state_dict(self, model: str) -> _state_dicts.GpuCachedStateDict:
        if model in self._wrapped_state_dicts:
            return self._wrapped_state_dicts[model]
        raw_state_dict = self._backing_provider.get_or_create_alias_state_dict(model)
        wrapped = _state_dicts.GpuCachedStateDict(
            raw_state_dict,
            config=self._cache_config,
            cache_name=model,
        )
        self._wrapped_state_dicts[model] = wrapped
        return wrapped

    def close(self) -> None:
        self.flush()
        self._backing_provider.close()

    def flush(self) -> None:
        for wrapped in self._wrapped_state_dicts.values():
            wrapped.flush()

    def save_output(
        self,
        plan: SurgeryPlan,
        *,
        default_shard_size: str,
        max_io_workers: int,
    ) -> Path:
        self.flush()
        return self._backing_provider.save_output(
            plan,
            default_shard_size=default_shard_size,
            max_io_workers=max_io_workers,
        )


def wrap_provider_with_gpu_cache(
    provider: BaseStateDictProvider,
    *,
    cache_config: _state_dicts.GpuCacheConfig,
) -> GpuCachedStateDictProvider:
    return GpuCachedStateDictProvider(
        provider,
        cache_config=cache_config,
    )


assert callable(wrap_provider_with_gpu_cache)


def create_state_dict_provider(
    *,
    provider: str,
    model_paths: dict[str, Path],
    max_io_workers: int,
    arena_root: Path,
    arena_segment_size: str,
) -> BaseStateDictProvider:
    provider_name = provider.strip().lower()

    if provider_name == "inmemory":
        return InMemoryStateDictProvider(
            model_paths,
            max_io_workers=max_io_workers,
        )

    if provider_name == "arena":
        segment_size_bytes = parse_shard_size(arena_segment_size)
        if segment_size_bytes is None:
            raise ProviderError("arena-segment-size must not be 'none'")

        arena = _SegmentedFileBackedArena(
            arena_root,
            segment_size_bytes=segment_size_bytes,
        )
        return ArenaStateDictProvider(
            model_paths,
            arena=arena,
            max_io_workers=max_io_workers,
        )

    raise ProviderError("provider must be either 'inmemory' or 'arena'")


__all__ = [
    "BaseStateDictProvider",
    "InMemoryStateDictProvider",
    "ArenaStateDictProvider",
    "GpuCachedStateDictProvider",
    "create_state_dict_provider",
    "wrap_provider_with_gpu_cache",
]
