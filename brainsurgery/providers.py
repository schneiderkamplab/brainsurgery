from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file as save_safetensors_file

from .arena import ArenaError, SegmentedFileBackedArena, TensorSlot
from .model import (
    load_state_dict_from_path,
    resolve_output_destination,
    resolve_sharded_output_directory,
    tqdm,
    save_sharded_safetensors,
)
from .plan import SurgeryPlan
from .transform import StateDictLike, infer_output_model

logger = logging.getLogger("brainsurgery")


# ============================================================
# State-dict implementations
# ============================================================


class InMemoryStateDict(StateDictLike):
    def __init__(self):
        self._data: Dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ArenaError(f"value for key {key!r} is not a tensor")
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def slot(self, key: str) -> torch.Tensor:
        return self._data[key]

    def bind_slot(self, key: str, slot: torch.Tensor) -> None:
        if not torch.is_tensor(slot):
            raise ArenaError(f"slot for key {key!r} is not a tensor")
        self._data[key] = slot

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()


class ArenaStateDict(StateDictLike):
    def __init__(self, arena: SegmentedFileBackedArena):
        self._arena = arena
        self._slots: Dict[str, TensorSlot] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            slot = self._slots[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        return self._arena.tensor_from_slot(slot)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ArenaError(f"value for key {key!r} is not a tensor")
        self._slots[key] = self._arena.store_tensor(value)

    def __delitem__(self, key: str) -> None:
        del self._slots[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def slot(self, key: str) -> TensorSlot:
        try:
            return self._slots[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def bind_slot(self, key: str, slot: TensorSlot) -> None:
        if not isinstance(slot, TensorSlot):
            raise ArenaError(f"slot for key {key!r} is not a TensorSlot")
        self._slots[key] = slot

    def keys(self):
        return self._slots.keys()

    def items(self):
        for key in self._slots:
            yield key, self[key]

    def values(self):
        for key in self._slots:
            yield self[key]


# ============================================================
# Providers
# ============================================================


class BaseStateDictProvider:
    def __init__(self, model_paths: Dict[str, Path], max_io_workers: int):
        self.model_paths = model_paths
        self.max_io_workers = max_io_workers
        self.state_dicts: Dict[str, StateDictLike] = {}

    def get_state_dict(self, model: str) -> StateDictLike:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def save_output(
        self,
        plan: SurgeryPlan,
        *,
        default_shard_size: str,
        max_io_workers: int,
    ) -> Path:
        output_model = infer_output_model(plan)
        state_dict = self.get_state_dict(output_model)

        output_path, output_format, shard_size = resolve_output_destination(
            plan.output,
            default_shard_size=default_shard_size,
        )

        logger.info(
            "Closing incision and preserving brain '%s' to %s (%s)",
            output_model,
            output_path,
            output_format,
        )

        if output_format == "torch":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict.items()), output_path)
            logger.info("Patient stable. Wrote %d tensors to %s", len(state_dict), output_path)
            return output_path

        if shard_size is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_safetensors_file(dict(state_dict.items()), str(output_path))
            logger.info("Patient stable. Wrote %d tensors to %s", len(state_dict), output_path)
            return output_path

        output_dir = resolve_sharded_output_directory(plan.output.path, output_path)
        index_path = save_sharded_safetensors(
            dict(state_dict.items()),
            output_dir,
            shard_size,
            max_io_workers=max_io_workers,
        )
        logger.info(
            "Patient stable. Wrote %d tensors across sharded safetensors in %s",
            len(state_dict),
            output_dir,
        )
        return index_path


class InMemoryStateDictProvider(BaseStateDictProvider):
    def __init__(self, model_paths: Dict[str, Path], max_io_workers: int):
        super().__init__(model_paths, max_io_workers=max_io_workers)

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        if model not in self.state_dicts:
            path = self.model_paths[model]
            logger.info("Opening cranium for brain '%s' at %s", model, path)

            loaded = load_state_dict_from_path(path, max_io_workers=self.max_io_workers)
            sd = InMemoryStateDict()
            for key, tensor in tqdm(loaded.items(), desc=f"Loading brain '{model}'", unit="tensor"):
                sd[key] = tensor
            del loaded

            self.state_dicts[model] = sd
            logger.info(
                "Brain '%s' exposed: %d tensors on the operating table",
                model,
                len(sd),
            )

        state_dict = self.state_dicts[model]
        assert isinstance(state_dict, InMemoryStateDict)
        return state_dict


class ArenaStateDictProvider(BaseStateDictProvider):
    def __init__(
        self,
        model_paths: Dict[str, Path],
        *,
        arena: SegmentedFileBackedArena,
        max_io_workers: int,
    ):
        super().__init__(model_paths, max_io_workers=max_io_workers)
        self.arena = arena

    def close(self) -> None:
        self.arena.close()

    def get_state_dict(self, model: str) -> ArenaStateDict:
        if model not in self.state_dicts:
            path = self.model_paths[model]
            logger.info("Opening cranium for brain '%s' at %s", model, path)

            loaded = load_state_dict_from_path(path, max_io_workers=self.max_io_workers)
            sd = ArenaStateDict(self.arena)
            for key, tensor in tqdm(loaded.items(), desc=f"Loading brain '{model}'", unit="tensor"):
                sd[key] = tensor
            del loaded

            self.state_dicts[model] = sd
            logger.info(
                "Brain '%s' exposed in arena: %d tensors on the operating table",
                model,
                len(sd),
            )

        state_dict = self.state_dicts[model]
        assert isinstance(state_dict, ArenaStateDict)
        return state_dict
