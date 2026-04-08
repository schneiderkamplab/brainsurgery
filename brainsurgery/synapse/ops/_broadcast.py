from __future__ import annotations

from typing import Any


def _normalize_dim_token(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


def _is_dim_one(value: Any) -> bool:
    return _normalize_dim_token(value) == 1


def broadcast_dim(left: Any, right: Any) -> Any | None:
    left_norm = _normalize_dim_token(left)
    right_norm = _normalize_dim_token(right)
    if left_norm == right_norm:
        return left_norm
    if _is_dim_one(left_norm):
        return right_norm
    if _is_dim_one(right_norm):
        return left_norm
    return None


def broadcast_shape(
    left: tuple[Any, ...] | None,
    right: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    if left is None:
        return right
    if right is None:
        return left
    max_rank = max(len(left), len(right))
    left_full = (1,) * (max_rank - len(left)) + left
    right_full = (1,) * (max_rank - len(right)) + right
    result: list[Any] = []
    for left_dim, right_dim in zip(left_full, right_full, strict=True):
        merged = broadcast_dim(left_dim, right_dim)
        if merged is None:
            return None
        result.append(merged)
    return tuple(result)


def broadcast_last_dim(left: Any, right: Any) -> Any | None:
    if left is None:
        return _normalize_dim_token(right)
    if right is None:
        return _normalize_dim_token(left)
    return broadcast_dim(left, right)


__all__ = ["broadcast_dim", "broadcast_last_dim", "broadcast_shape"]
