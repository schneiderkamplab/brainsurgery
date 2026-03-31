from __future__ import annotations

from .emitter import PyTorchEmitter

EMITTER_CLASS = PyTorchEmitter
BACKEND_NAME = "pytorch"

__all__ = ["PyTorchEmitter"]
