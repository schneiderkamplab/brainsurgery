from __future__ import annotations

import pytest
import torch

from brainsurgery.synapse import lower_axon_program_to_graph_ir
from brainsurgery.synapse.axon import (
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    normalize_closed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.codegen2_jax import (
    SUPPORTED_JAX_PRIMITIVES,
)
from brainsurgery.synapse.axon.codegen2_jax import (
    emit_model_code_from_graph_ir as emit_jax_model_code,
)
from brainsurgery.synapse.axon.codegen2_mlx import (
    NON_OBVIOUS_MLX_OPS,
    SUPPORTED_MLX_PRIMITIVES,
)
from brainsurgery.synapse.axon.codegen2_mlx import (
    emit_model_code_from_graph_ir as emit_mlx_model_code,
)
from brainsurgery.synapse.axon.codegen2_tinygrad import (
    NON_OBVIOUS_TINYGRAD_OPS,
    SUPPORTED_TINYGRAD_PRIMITIVES,
)
from brainsurgery.synapse.axon.codegen2_tinygrad import (
    emit_model_code_from_graph_ir as emit_tinygrad_model_code,
)
from brainsurgery.synapse.axon.codegen2_torch import Codegen2GraphModel


def _lower(path):  # type: ignore[no-untyped-def]
    resolved = resolve_axon_program_from_path(path).ast
    normalized = normalize_closed_axon_file(resolved, main_module="main")
    elaborated = elaborate_closed_axon_file(normalized, main_module="main")
    flat = flatten_closed_axon_file(elaborated, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    return lower_axon_program_to_graph_ir(typed, main_module="main")


def test_keyed_random_stable_argsort_and_scatter_reduce_execute(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "main.axon"
    source.write_text(
        """
import Tensor (IdxTensor, argsort, scatter_reduce)
import Random (key, normal)

{-# MAIN "main" #-}

main :: Tensor[2,3] -> IdxTensor[2,3] -> Tensor[2,3] -> (IdxTensor[2,3], Tensor[2,3], Tensor[2,3], Tensor[2,3], Int, Int)
main x index src = do
  order <- argsort x dim=-1 descending=false stable=true
  reduced <- scatter_reduce x index src dim=-1 reduce="sum" include_self=false
  random1, next1 <- normal (key 7) x [2, 3]
  random2, next2 <- normal (key 7) x [2, 3]
  return order, reduced, random1, random2, next1, next2
""",
        encoding="utf-8",
    )
    model = Codegen2GraphModel.from_state_dict({}, graph=_lower(source))
    x = torch.tensor([[2.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
    index = torch.tensor([[0, 0, 2], [1, 1, 0]])
    src = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    outputs = model.forward(x=x, index=index, src=src)
    order, reduced, random1, random2, next1, next2 = outputs.values()

    assert torch.equal(order, torch.tensor([[1, 2, 0], [2, 1, 0]]))
    assert torch.equal(reduced, torch.tensor([[7.0, 1.0, 5.0], [8.0, 13.0, 1.0]]))
    assert torch.equal(random1, random2)
    assert next1 == next2 == 8


def test_acos_executes_for_tensor_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "main.axon"
    source.write_text(
        """
import Math (acos)

{-# MAIN "main" #-}

main :: Tensor[3] -> Tensor[3]
main x = acos x
""",
        encoding="utf-8",
    )
    model = Codegen2GraphModel.from_state_dict({}, graph=_lower(source))
    x = torch.tensor([-1.0, 0.0, 1.0])

    output = model.forward(x=x)
    if isinstance(output, dict):
        output = next(iter(output.values()))

    assert torch.allclose(output, torch.acos(x))


def test_keyed_uniform_is_reproducible_and_bounded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "main.axon"
    source.write_text(
        """
import Random (key, uniform)

{-# MAIN "main" #-}

main :: Tensor[2,3] -> (Tensor[2,3], Tensor[2,3], Int, Int)
main ref = do
  sample1, next1 <- uniform (key 11) ref [2, 3]
  sample2, next2 <- uniform (key 11) ref [2, 3]
  return sample1, sample2, next1, next2
""",
        encoding="utf-8",
    )
    model = Codegen2GraphModel.from_state_dict({}, graph=_lower(source))
    outputs = model.forward(ref=torch.zeros((2, 3)))
    sample1, sample2, next1, next2 = outputs.values()

    assert torch.equal(sample1, sample2)
    assert torch.all(sample1 >= 0)
    assert torch.all(sample1 < 1)
    assert next1 == next2 == 12


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("sum", [[7.0, 1.0, 5.0]]),
        ("prod", [[12.0, 1.0, 5.0]]),
        ("mean", [[3.5, 1.0, 5.0]]),
        ("max", [[4.0, 1.0, 5.0]]),
        ("min", [[3.0, 1.0, 5.0]]),
        ("amax", [[4.0, 1.0, 5.0]]),
        ("amin", [[3.0, 1.0, 5.0]]),
    ],
)
def test_scatter_reduce_modes_execute(
    tmp_path, reduction: str, expected: list[list[float]]
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / f"scatter_{reduction}.axon"
    source.write_text(
        f"""
import Tensor (IdxTensor, scatter_reduce)

{{-# MAIN "main" #-}}

main :: Tensor[1,3] -> IdxTensor[1,3] -> Tensor[1,3] -> Tensor[1,3]
main x index src = scatter_reduce x index src dim=-1 reduce="{reduction}" include_self=false
""",
        encoding="utf-8",
    )
    model = Codegen2GraphModel.from_state_dict({}, graph=_lower(source))
    output = model.forward(
        x=torch.tensor([[9.0, 1.0, 5.0]]),
        index=torch.tensor([[0, 0, 2]]),
        src=torch.tensor([[3.0, 4.0, 5.0]]),
    )
    if isinstance(output, dict):
        output = next(iter(output.values()))

    assert torch.equal(output, torch.tensor(expected))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            'main x = Tensor.argsort x dim=2 descending=false stable=true',
            "argsort dimension 2 is out of range",
        ),
        (
            'main x index src = Tensor.scatter_reduce x index src dim=-1 reduce="median" include_self=false',
            "unsupported scatter_reduce reduction",
        ),
        (
            'main x index src = Tensor.scatter_reduce x index src dim=-3 reduce="sum" include_self=false',
            "scatter_reduce dimension -3 is out of range",
        ),
    ],
)
def test_sort_and_scatter_contracts_reject_invalid_literals(
    tmp_path, body: str, message: str
) -> None:  # type: ignore[no-untyped-def]
    args = "x" if "argsort" in body else "x index src"
    signature = (
        "Tensor[2,3] -> IdxTensor[2,3]"
        if "argsort" in body
        else "Tensor[2,3] -> IdxTensor[2,3] -> Tensor[2,3] -> Tensor[2,3]"
    )
    source = tmp_path / "invalid.axon"
    source.write_text(
        f"""
import Tensor (IdxTensor)

{{-# MAIN "main" #-}}

main :: {signature}
main {args} = {body.split(' = ', 1)[1]}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _lower(source)


def test_acos_rejects_scalar_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "scalar_acos.axon"
    source.write_text(
        """
import Math (acos)

{-# MAIN "main" #-}

main :: Float -> Float
main x = acos x
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        _lower(source)


def test_new_primitive_backend_support_is_explicit() -> None:
    all_new_ops = {
        "acos",
        "round",
        "argsort",
        "scatter_reduce",
        "random_normal",
        "random_uniform",
    }
    assert all_new_ops <= SUPPORTED_JAX_PRIMITIVES

    assert {"acos", "round", "argsort", "random_normal", "random_uniform"} <= (
        SUPPORTED_MLX_PRIMITIVES
    )
    assert "scatter_reduce" not in SUPPORTED_MLX_PRIMITIVES
    assert "scatter_reduce" in NON_OBVIOUS_MLX_OPS

    assert {"acos", "round", "scatter_reduce"} <= SUPPORTED_TINYGRAD_PRIMITIVES
    assert {"argsort", "random_normal", "random_uniform"}.isdisjoint(
        SUPPORTED_TINYGRAD_PRIMITIVES
    )
    assert {"argsort", "random_normal", "random_uniform"} <= (
        NON_OBVIOUS_TINYGRAD_OPS.keys()
    )


def test_new_primitive_backend_lowerings_emit_valid_python(tmp_path) -> None:  # type: ignore[no-untyped-def]
    common_source = tmp_path / "common.axon"
    common_source.write_text(
        """
import Math (acos)
import Random (key, normal, uniform)
import Tensor (IdxTensor, argsort)

{-# MAIN "main" #-}

main :: Tensor[2,3] -> (IdxTensor[2,3], Tensor[2,3], Tensor[2,3], Tensor[2,3])
main x = do
  order <- argsort x dim=-1 descending=true stable=true
  angle <- acos x
  gaussian, _ <- normal (key 3) x [2, 3]
  bounded, _ <- uniform (key 5) x [2, 3]
  return order, angle, gaussian, bounded
""",
        encoding="utf-8",
    )
    common_graph = _lower(common_source)
    jax_code = emit_jax_model_code(common_graph)
    mlx_code = emit_mlx_model_code(common_graph)
    compile(jax_code, "<generated-jax>", "exec")
    compile(mlx_code, "<generated-mlx>", "exec")
    assert "jax.random.normal" in jax_code
    assert "jnp.argsort" in jax_code
    assert "mx.random.normal" in mlx_code
    assert "mx.argsort" in mlx_code

    tinygrad_source = tmp_path / "tinygrad.axon"
    tinygrad_source.write_text(
        """
import Math (acos)
import Tensor (IdxTensor, scatter_reduce)

{-# MAIN "main" #-}

main :: Tensor[2,3] -> IdxTensor[2,3] -> Tensor[2,3] -> (Tensor[2,3], Tensor[2,3])
main x index src = do
  angle <- acos x
  reduced <- scatter_reduce x index src dim=-1 reduce="max" include_self=false
  return angle, reduced
""",
        encoding="utf-8",
    )
    tinygrad_code = emit_tinygrad_model_code(_lower(tinygrad_source))
    compile(tinygrad_code, "<generated-tinygrad>", "exec")
    assert ".acos()" in tinygrad_code
    assert ".scatter_reduce(" in tinygrad_code


def test_jax_scatter_reduce_mean_handles_duplicate_indices(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jnp = pytest.importorskip("jax.numpy")
    source = tmp_path / "jax_scatter_mean.axon"
    source.write_text(
        """
import Tensor (IdxTensor, scatter_reduce)

{-# MAIN "main" #-}

main :: Tensor[1,3] -> IdxTensor[1,3] -> Tensor[1,3] -> Tensor[1,3]
main x index src = scatter_reduce x index src dim=-1 reduce="mean" include_self=false
""",
        encoding="utf-8",
    )
    namespace: dict[str, object] = {}
    exec(emit_jax_model_code(_lower(source)), namespace)
    model = namespace["AxonJaxModel"].from_state_dict({})
    output = model.forward(
        x=jnp.array([[9.0, 1.0, 5.0]]),
        index=jnp.array([[0, 0, 2]]),
        src=jnp.array([[3.0, 4.0, 5.0]]),
    )
    if isinstance(output, dict):
        output = next(iter(output.values()))

    assert torch.equal(
        torch.tensor(jnp.asarray(output).tolist()), torch.tensor([[3.5, 1.0, 5.0]])
    )
