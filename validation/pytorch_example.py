from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import torch

AliasStore = dict[str, dict[str, torch.Tensor]]

@dataclass
class RuntimeFlags:
	dry_run: bool = False
	preview: bool = False
	verbose: bool = False

@dataclass(frozen=True)
class TensorRef:
	alias: str
	name: str
	slice_spec: str | None = None

def _parse_tensor_ref(ref: str, *, default_alias: str = "model") -> TensorRef:
	if not isinstance(ref, str) or not ref:
		raise ValueError("reference must be a non-empty string")

	base = ref
	slice_spec: str | None = None
	slice_index = ref.rfind("::[")
	if slice_index != -1:
		base = ref[:slice_index]
		slice_spec = ref[slice_index + 2 :]

	if "::" in base:
		alias, name = base.split("::", 1)
		if not alias or not name:
			raise ValueError(f"invalid tensor reference: {ref}")
	else:
		alias, name = default_alias, base

	return TensorRef(alias=alias, name=name, slice_spec=slice_spec)


def _slice_parts(text: str) -> tuple[slice | int, ...]:
	if not (text.startswith("[") and text.endswith("]")):
		raise ValueError(f"invalid slice syntax: {text}")
	inner = text[1:-1].strip()
	if not inner:
		return (slice(None),)

	out: list[slice | int] = []
	for raw in inner.split(","):
		token = raw.strip()
		if not token:
			out.append(slice(None))
			continue
		if ":" in token:
			fields = token.split(":")
			if len(fields) > 3:
				raise ValueError(f"invalid slice token: {token}")
			vals: list[int | None] = []
			for field in fields:
				stripped = field.strip()
				vals.append(None if stripped == "" else int(stripped))
			while len(vals) < 3:
				vals.append(None)
			out.append(slice(vals[0], vals[1], vals[2]))
		else:
			out.append(int(token))
	return tuple(out)


def _get_tensor_view(store: AliasStore, ref: str, *, default_alias: str = "model") -> torch.Tensor:
	parsed = _parse_tensor_ref(ref, default_alias=default_alias)
	tensor = store[parsed.alias][parsed.name]
	if parsed.slice_spec is None:
		return tensor
	return tensor[_slice_parts(parsed.slice_spec)]


def _set_tensor(
	store: AliasStore,
	ref: str,
	tensor: torch.Tensor,
	*,
	default_alias: str = "model",
	must_exist: bool | None = None,
) -> None:
	parsed = _parse_tensor_ref(ref, default_alias=default_alias)
	if parsed.slice_spec is not None:
		raise ValueError("destination references must not be sliced")
	alias_map = store.setdefault(parsed.alias, {})
	exists = parsed.name in alias_map
	if must_exist is True and not exists:
		raise KeyError(f"destination missing: {parsed.alias}::{parsed.name}")
	if must_exist is False and exists:
		raise KeyError(f"destination already exists: {parsed.alias}::{parsed.name}")
	alias_map[parsed.name] = tensor


def _resolve_pattern(target: str, *, default_alias: str = "model") -> tuple[str, re.Pattern[str]]:
	if "::" in target:
		alias, expr = target.split("::", 1)
	else:
		alias, expr = default_alias, target
	return alias, re.compile(expr)


# set
def transform_set(
	flags: RuntimeFlags,
	*,
	dry_run: bool | None = None,
	preview: bool | None = None,
	verbose: bool | None = None,
) -> RuntimeFlags:
	if dry_run is not None:
		flags.dry_run = dry_run
	if preview is not None:
		flags.preview = preview
	if verbose is not None:
		flags.verbose = verbose
	return flags


# help
def transform_help_diff() -> str:
	return "Compare two tensor sets by names and values, reporting missing and differing keys."


# dump
def transform_dump_compact(store: AliasStore, *, alias: str = "model", limit: int = 16) -> list[str]:
	lines: list[str] = []
	for name in sorted(store.get(alias, {}))[:limit]:
		t = store[alias][name]
		lines.append(f"{alias}::{name} shape={tuple(t.shape)} dtype={t.dtype}")
	return lines


# prefixes
def transform_prefixes_list(store: AliasStore) -> list[str]:
	return sorted(store.keys())


def transform_prefixes_add(store: AliasStore, alias: str) -> None:
	if alias in store:
		raise KeyError(f"alias already exists: {alias}")
	store[alias] = {}


def transform_prefixes_remove(store: AliasStore, alias: str) -> None:
	del store[alias]


def transform_prefixes_rename(store: AliasStore, src: str, dst: str) -> None:
	if src == dst:
		raise ValueError("source and destination aliases must differ")
	if dst in store:
		raise KeyError(f"alias already exists: {dst}")
	store[dst] = store.pop(src)


# init tensors
def transform_zeroes(store: AliasStore, target: str, shape: tuple[int, ...]) -> None:
	_set_tensor(store, target, torch.zeros(shape, dtype=torch.float32), must_exist=False)


def transform_ones(store: AliasStore, target: str, shape: tuple[int, ...]) -> None:
	_set_tensor(store, target, torch.ones(shape, dtype=torch.float32), must_exist=False)


def transform_rand(
	store: AliasStore,
	target: str,
	shape: tuple[int, ...],
	*,
	distribution: str = "uniform",
	low: float = 0.0,
	high: float = 1.0,
	mean: float = 0.0,
	std: float = 1.0,
	seed: int | None = None,
) -> None:
	if seed is not None:
		g = torch.Generator()
		g.manual_seed(seed)
	else:
		g = None
	out = torch.empty(shape, dtype=torch.float32)
	if distribution == "uniform":
		out.uniform_(low, high, generator=g)
	elif distribution == "normal":
		out.normal_(mean, std, generator=g)
	else:
		raise ValueError("distribution must be 'uniform' or 'normal'")
	_set_tensor(store, target, out, must_exist=False)


# assert
def assert_exists(store: AliasStore, ref: str) -> None:
	parsed = _parse_tensor_ref(ref)
	if parsed.name not in store.get(parsed.alias, {}):
		raise AssertionError(f"missing tensor: {parsed.alias}::{parsed.name}")


def assert_not_exists(store: AliasStore, ref: str) -> None:
	parsed = _parse_tensor_ref(ref)
	if parsed.name in store.get(parsed.alias, {}):
		raise AssertionError(f"tensor unexpectedly exists: {parsed.alias}::{parsed.name}")


def assert_shape(store: AliasStore, ref: str, expected: tuple[int, ...]) -> None:
	actual = tuple(_get_tensor_view(store, ref).shape)
	if actual != expected:
		raise AssertionError(f"shape mismatch: {actual} != {expected}")


def assert_iszero(store: AliasStore, ref: str) -> None:
	if not bool(torch.all(_get_tensor_view(store, ref) == 0).item()):
		raise AssertionError(f"tensor is not all zeros: {ref}")


def assert_not_iszero(store: AliasStore, ref: str) -> None:
	if bool(torch.all(_get_tensor_view(store, ref) == 0).item()):
		raise AssertionError(f"tensor is all zeros: {ref}")


def assert_dtype(store: AliasStore, ref: str, dtype: torch.dtype) -> None:
	actual = _get_tensor_view(store, ref).dtype
	if actual != dtype:
		raise AssertionError(f"dtype mismatch: {actual} != {dtype}")


def assert_equal(store: AliasStore, left: str, right: str, *, eps: float | None = None) -> None:
	a = _get_tensor_view(store, left)
	b = _get_tensor_view(store, right)
	if a.shape != b.shape:
		raise AssertionError(f"shape mismatch: {a.shape} != {b.shape}")
	if a.dtype != b.dtype:
		raise AssertionError(f"dtype mismatch: {a.dtype} != {b.dtype}")
	if eps is None:
		ok = torch.equal(a, b)
	else:
		ok = bool(torch.allclose(a, b, atol=eps, rtol=0.0))
	if not ok:
		raise AssertionError(f"equal failed: {left} != {right}")


def assert_equal_regex_map(
	store: AliasStore,
	left: str,
	right_template: str,
	*,
	default_right_alias: str = "model",
	eps: float | None = None,
) -> None:
	left_alias, left_re = _resolve_pattern(left)
	for name in sorted(store[left_alias].keys()):
		match = left_re.fullmatch(name)
		if match is None:
			continue
		expanded = match.expand(right_template)
		if "::" in expanded:
			right_ref = expanded
		else:
			right_ref = f"{default_right_alias}::{expanded}"
		assert_equal(store, f"{left_alias}::{name}", right_ref, eps=eps)


# copy/move/delete/assign
def transform_copy(store: AliasStore, src: str, dst: str) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, src).clone(), must_exist=False)


def transform_move(store: AliasStore, src: str, dst: str, *, default_alias: str = "model") -> None:
	src_ref = _parse_tensor_ref(src, default_alias=default_alias)
	dst_ref = _parse_tensor_ref(dst, default_alias=default_alias)
	if src_ref.slice_spec is not None or dst_ref.slice_spec is not None:
		raise ValueError("move does not support sliced source or destination")
	if dst_ref.name in store.setdefault(dst_ref.alias, {}):
		raise KeyError(f"destination already exists: {dst_ref.alias}::{dst_ref.name}")
	store[dst_ref.alias][dst_ref.name] = store[src_ref.alias].pop(src_ref.name)


def transform_delete(store: AliasStore, target: str, *, default_alias: str = "model") -> int:
	alias, pattern = _resolve_pattern(target, default_alias=default_alias)
	names = [name for name in list(store.get(alias, {})) if pattern.fullmatch(name)]
	for name in names:
		del store[alias][name]
	return len(names)


def transform_assign(store: AliasStore, src: str, dst: str) -> None:
	src_t = _get_tensor_view(store, src)
	dst_ref = _parse_tensor_ref(dst)
	if dst_ref.slice_spec is not None:
		raise ValueError("assign destination must not be sliced")
	dst_t = store[dst_ref.alias][dst_ref.name]
	if src_t.shape != dst_t.shape:
		raise ValueError("assign requires identical shapes")
	if src_t.dtype != dst_t.dtype:
		raise ValueError("assign requires identical dtypes")
	dst_t.copy_(src_t)


# arithmetic
def transform_add(store: AliasStore, from_a: str, from_b: str, to: str) -> None:
	dst = _get_tensor_view(store, to)
	dst.copy_(_get_tensor_view(store, from_a) + _get_tensor_view(store, from_b))


def transform_subtract(store: AliasStore, from_a: str, from_b: str, to: str) -> None:
	dst = _get_tensor_view(store, to)
	dst.copy_(_get_tensor_view(store, from_a) - _get_tensor_view(store, from_b))


def transform_multiply(store: AliasStore, from_a: str, from_b: str, to: str) -> None:
	dst = _get_tensor_view(store, to)
	dst.copy_(_get_tensor_view(store, from_a) * _get_tensor_view(store, from_b))


def transform_add_(store: AliasStore, from_ref: str, to: str) -> None:
	_get_tensor_view(store, to).add_(_get_tensor_view(store, from_ref))


def transform_subtract_(store: AliasStore, from_ref: str, to: str) -> None:
	_get_tensor_view(store, to).sub_(_get_tensor_view(store, from_ref))


def transform_scale(store: AliasStore, src: str, dst: str, by: float) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, src) * by, must_exist=False)


def transform_scale_(store: AliasStore, target: str, by: float) -> None:
	_get_tensor_view(store, target).mul_(by)


# fill
def transform_fill_constant(store: AliasStore, src: str, dst: str, value: float) -> None:
	template = _get_tensor_view(store, src)
	_set_tensor(store, dst, torch.full_like(template, value), must_exist=False)


def transform_fill_rand(
	store: AliasStore,
	src: str,
	dst: str,
	*,
	distribution: str,
	seed: int | None = None,
	low: float = 0.0,
	high: float = 1.0,
	mean: float = 0.0,
	std: float = 1.0,
) -> None:
	template = _get_tensor_view(store, src)
	out = torch.empty_like(template)
	g = None
	if seed is not None:
		g = torch.Generator(device=out.device)
		g.manual_seed(seed)
	if distribution == "uniform":
		out.uniform_(low, high, generator=g)
	elif distribution == "normal":
		out.normal_(mean, std, generator=g)
	else:
		raise ValueError("distribution must be 'uniform' or 'normal'")
	_set_tensor(store, dst, out, must_exist=False)


def transform_fill_tensor_(store: AliasStore, target: str, values: Any) -> None:
	dst = _get_tensor_view(store, target)
	src = torch.as_tensor(values, dtype=dst.dtype, device=dst.device)
	dst.copy_(src.expand_as(dst))


# clamp/cast
def transform_clamp(store: AliasStore, src: str, dst: str, *, min_value: float, max_value: float) -> None:
	_set_tensor(
		store,
		dst,
		torch.clamp(_get_tensor_view(store, src), min=min_value, max=max_value),
		must_exist=False,
	)


def transform_clamp_(store: AliasStore, target: str, *, min_value: float, max_value: float) -> None:
	_get_tensor_view(store, target).clamp_(min=min_value, max=max_value)


def transform_cast(store: AliasStore, src: str, dst: str, *, dtype: torch.dtype) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, src).to(dtype=dtype), must_exist=False)


def transform_cast_(store: AliasStore, target: str, *, dtype: torch.dtype) -> None:
	parsed = _parse_tensor_ref(target)
	if parsed.slice_spec is not None:
		raise ValueError("cast_ target must not be sliced")
	store[parsed.alias][parsed.name] = store[parsed.alias][parsed.name].to(dtype=dtype)


# shape ops
def transform_split(
	store: AliasStore,
	src: str,
	dsts: list[str],
	sizes: list[int],
	*,
	dim: int = 0,
) -> None:
	chunks = torch.split(_get_tensor_view(store, src), sizes, dim=dim)
	if len(chunks) != len(dsts):
		raise ValueError("number of split outputs does not match destinations")
	for name, chunk in zip(dsts, chunks):
		_set_tensor(store, name, chunk.clone(), must_exist=False)


def transform_concat(store: AliasStore, srcs: list[str], dst: str, *, dim: int = 0) -> None:
	out = torch.cat([_get_tensor_view(store, ref) for ref in srcs], dim=dim)
	_set_tensor(store, dst, out, must_exist=False)


def transform_reshape(store: AliasStore, src: str, dst: str, shape: tuple[int, ...]) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, src).reshape(shape), must_exist=False)


def transform_reshape_(store: AliasStore, target: str, shape: tuple[int, ...]) -> None:
	parsed = _parse_tensor_ref(target)
	if parsed.slice_spec is not None:
		raise ValueError("reshape_ target must not be sliced")
	store[parsed.alias][parsed.name] = store[parsed.alias][parsed.name].reshape(shape)


def transform_permute(store: AliasStore, src: str, dst: str, order: tuple[int, ...]) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, src).permute(order), must_exist=False)


def transform_matmul(store: AliasStore, from_a: str, from_b: str, dst: str) -> None:
	_set_tensor(store, dst, _get_tensor_view(store, from_a) @ _get_tensor_view(store, from_b), must_exist=False)


# phlora
def _low_rank_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	if weight.ndim != 2:
		raise ValueError("phlora requires 2D tensors")
	m, n = weight.shape
	if rank <= 0:
		raise ValueError("rank must be positive")
	rank = min(rank, min(m, n))
	u, s, vh = torch.linalg.svd(weight, full_matrices=False)
	u_k = u[:, :rank]
	s_k = s[:rank]
	vh_k = vh[:rank, :]
	sqrt_s = s_k.sqrt()
	a = sqrt_s[:, None] * vh_k
	b = u_k * sqrt_s
	recon = (u_k * s_k) @ vh_k
	return recon, a, b


def transform_phlora_(store: AliasStore, target: str, rank: int) -> None:
	parsed = _parse_tensor_ref(target)
	if parsed.slice_spec is not None:
		raise ValueError("phlora_ target must not be sliced")
	weight = store[parsed.alias][parsed.name]
	recon, _a, _b = _low_rank_factors(weight, rank)
	store[parsed.alias][parsed.name] = recon


def transform_phlora(
	store: AliasStore,
	target: str,
	target_a: str,
	target_b: str,
	rank: int,
	*,
	delete_original: bool = True,
) -> None:
	weight = _get_tensor_view(store, target)
	_recon, a, b = _low_rank_factors(weight, rank)
	_set_tensor(store, target_a, a, must_exist=False)
	_set_tensor(store, target_b, b, must_exist=False)
	if delete_original:
		parsed = _parse_tensor_ref(target)
		del store[parsed.alias][parsed.name]


# save/load
def transform_save_tensor(store: AliasStore, target: str, path: str | Path) -> None:
	# Plain PyTorch equivalent to tool save: writes tensor with torch.save.
	torch.save(_get_tensor_view(store, target).cpu(), Path(path))


def transform_load_tensor(store: AliasStore, path: str | Path, to: str, *, map_location: str = "cpu") -> None:
	tensor = torch.load(Path(path), map_location=map_location)
	if not isinstance(tensor, torch.Tensor):
		raise TypeError("expected a tensor payload when loading single tensor")
	_set_tensor(store, to, tensor, must_exist=False)


def transform_load_alias_from_state_dict(store: AliasStore, path: str | Path, alias: str) -> None:
	payload = torch.load(Path(path), map_location="cpu")
	if not isinstance(payload, dict):
		raise TypeError("expected mapping payload for alias load")
	out: dict[str, torch.Tensor] = {}
	for key, value in payload.items():
		if isinstance(key, str) and isinstance(value, torch.Tensor):
			out[key] = value
	store[alias] = out


# diff
def transform_diff_aliases(
	store: AliasStore,
	left_alias: str,
	right_alias: str,
	*,
	eps: float | None = None,
) -> dict[str, list[str] | dict[str, str]]:
	left = store[left_alias]
	right = store[right_alias]
	left_names = set(left)
	right_names = set(right)

	missing_on_left = sorted(right_names - left_names)
	missing_on_right = sorted(left_names - right_names)
	differing: dict[str, str] = {}

	for name in sorted(left_names & right_names):
		a = left[name]
		b = right[name]
		if a.shape != b.shape:
			differing[name] = f"shape {tuple(a.shape)} != {tuple(b.shape)}"
			continue
		if a.dtype != b.dtype:
			differing[name] = f"dtype {a.dtype} != {b.dtype}"
			continue
		if eps is None:
			same = torch.equal(a, b)
		else:
			same = bool(torch.allclose(a, b, atol=eps, rtol=0.0))
		if not same:
			differing[name] = "values differ"

	return {
		"missing_on_left": missing_on_left,
		"missing_on_right": missing_on_right,
		"differing": differing,
	}


# execute
def transform_execute(store: AliasStore, steps: Iterable[Callable[[AliasStore], None]]) -> None:
	for step in steps:
		step(store)


# exit
def transform_exit() -> bool:
	return True
