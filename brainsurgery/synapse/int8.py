"""Dequant-on-load for W8A16 int8 checkpoints (opf-style export layout).

The quantizer stores each quantized parameter as
``<base>.int8`` (torch.int8) plus ``<base>.scale`` (fp32, symmetric
per-output-channel absmax reducing over dim 1), e.g.::

    block.0.mlp.swiglu.weight.int8   [E, in, out]  int8
    block.0.mlp.swiglu.weight.scale  [E, out]      fp32
    embedding.weight.int8            [V, D]        int8
    embedding.weight.scale           [V]           fp32

``materialize_int8_aliases`` rebuilds ``<base>`` as
``q.float() * scale.unsqueeze(1)`` cast to bf16 — bit-identical to the
published ``-dequant`` bf16 checkpoints emitted by the same quantizer.
"""

from __future__ import annotations

import torch

_INT8_SUFFIX = ".int8"
_SCALE_SUFFIX = ".scale"


def materialize_int8_aliases(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    drop_packed: bool = True,
) -> dict[str, torch.Tensor]:
    for key in list(state_dict.keys()):
        name = str(key)
        if not name.endswith(_INT8_SUFFIX):
            continue
        base = name[: -len(_INT8_SUFFIX)]
        scale_key = f"{base}{_SCALE_SUFFIX}"
        if scale_key not in state_dict:
            continue
        if torch.is_tensor(state_dict.get(base)):
            continue
        q = state_dict[key]
        scale = state_dict[scale_key]
        dequantized = q.float() * scale.float().unsqueeze(1)
        state_dict[base] = dequantized.to(dtype)
        if drop_packed:
            state_dict.pop(key, None)
            state_dict.pop(scale_key, None)
    return state_dict


__all__ = ["materialize_int8_aliases"]
