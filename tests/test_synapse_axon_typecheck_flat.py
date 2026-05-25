from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonExprCall,
    AxonExprInt,
    AxonExprName,
    AxonFile,
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    Constraint,
    TypeAliasDef,
    TypeDim,
    TypeFloat,
    TypeInt,
    TypeNamed,
    TypeTensor,
    TypeTuple,
    TypeVar,
)
from brainsurgery.synapse.axon.ast.render import render_axon_file
from brainsurgery.synapse.axon.elaborate import elaborate_closed_axon_file
from brainsurgery.synapse.axon.flatten import flatten_closed_axon_file
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.typecheck_shared import _TcCtx, _is_generic_named_type, _scoped_typevars
from brainsurgery.synapse.axon.typecheck2 import typecheck2_flat_axon_file
from brainsurgery.synapse.axon.lowering import lower_axon_program_to_graph_ir
from brainsurgery.synapse.axon.codegen2_torch import make_runtime2_model_class
from brainsurgery.synapse.axon.validate import validate_typed_axon_file


def _flat(program, *, main_module: str):
    normalized = normalize_closed_axon_file(program, main_module=main_module)
    elaborated = elaborate_closed_axon_file(normalized, main_module=main_module)
    return flatten_closed_axon_file(elaborated, main_module=main_module)


def _resolve_attention_wrapper(tmp_path: Path):
    source = """
import Attention (attention)

main :: Tensor[B,H,Q,HD] -> Tensor[B,H,K,HD] -> Tensor[B,H,K,HD] -> Tensor[B,1,Q,K] -> Tensor[B,H,Q,HD]
main q k v keep = attention q k v keep
"""
    path = tmp_path / "attention_main.axon"
    path.write_text(source)
    return resolve_axon_program_from_path(path).ast


def test_typecheck_flat_narrows_generated_loop_helper_signature() -> None:
    source = """
L = 4

step :: Tensor[B,S,D] -> Int -> Tensor[B,S,D]
step x i = x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  x <- for@h i <- [0..L) do
    x <- step x i
  return x
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    repeat = next(stmt for stmt in module.statements if isinstance(stmt, AxonRepeat))
    assert repeat.name is None
    assert repeat.from_expr.inferred_type is not None
    assert repeat.to_expr.inferred_type is not None
    assert repeat.step_expr.inferred_type is not None
    helper = next(module for module in typed.modules if module.name.startswith("main__loop_h_step_"))
    assert isinstance(helper.params[0].type_expr, TypeInt)
    assert not any(module.name.startswith("main__loop_h_recur_") for module in typed.modules)


def test_typecheck_flat_populates_expr_annotations() -> None:
    source = """
f :: Int -> Int
f x = x

main :: Int -> Int
main x = do
  y <- f x
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    bind_stmt = next(stmt for stmt in module.statements if isinstance(stmt, AxonBind))
    assert bind_stmt.expr.inferred_type is not None
    assert bind_stmt.expr.inferred_arity == 1
    repeat_stmts = [stmt for stmt in module.statements if isinstance(stmt, AxonRepeat)]
    assert repeat_stmts == []


def test_typecheck2_float_dim_arithmetic_stays_float() -> None:
    source = """
D :: Dim
D = 128

SCALE = 1.0 / D

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- x * SCALE
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    scale = next(module for module in typed.modules if module.name == "SCALE")
    assert isinstance(scale.return_type_expr, TypeFloat)
    main = next(module for module in typed.modules if module.name == "main")
    bind_stmt = next(
        stmt
        for stmt in main.statements
        if isinstance(stmt, AxonBind) and stmt.targets == ("y",)
    )
    assert isinstance(bind_stmt.expr.inferred_type, TypeTensor)


def test_typecheck2_dim_dim_arithmetic_stays_dim() -> None:
    source = """
D :: Dim
D = 128

H :: Dim
H = 8

WIDTH = D / H

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- x + WIDTH
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    width = next(module for module in typed.modules if module.name == "WIDTH")
    assert isinstance(width.return_type_expr, TypeDim)


def test_typecheck_flat_is_rooted_at_selected_main_module() -> None:
    source = """
main :: Int
main = do
  x <- 1
  return x
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    bad = AxonDefinition(
        name="bad",
        path_param=None,
        params=(),
        returns=(),
        statements=(
            AxonBind(
                targets=("x",),
                expr=AxonExprCall(callee="missing", args=(AxonExprInt(1),), kwargs={}),
            ),
            AxonReturn(values=(AxonExprName("x"),)),
        ),
        return_type_expr=TypeInt(),
    )
    flat = AxonFile(
        modules=(*flat.modules, bad),
        imports=flat.imports,
        imported_members=flat.imported_members,
        exports=flat.exports,
        pragmas=flat.pragmas,
        type_aliases=flat.type_aliases,
        origin_path=flat.origin_path,
    )
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    assert [module.name for module in typed.modules] == ["main"]


def test_typecheck_flat_unifies_embedding_dim_from_add() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast
    flat = _flat(resolved, main_module="gpt2")
    typed = typecheck2_flat_axon_file(flat, main_module="gpt2")
    validate_typed_axon_file(typed, main_module="gpt2")
    text = render_axon_file(typed, show_types=True)
    pos_line = next(
        line for line in text.splitlines() if "pos <-" in line and "NN.embedding" in line and "@@wpe" in line
    )
    layernorm_line = next(line for line in text.splitlines() if "NN.layernorm" in line and "@@ln_f" in line)
    assert "MODEL_DIM" in pos_line
    assert ",D]" not in pos_line
    assert "Tensor[" in layernorm_line and ",MODEL_DIM]" in layernorm_line


def test_typecheck_flat_exposes_signature_dim_as_term_dim() -> None:
    source = """
f :: Tensor[B,S,D] -> Int
f x = D
"""
    flat = _flat(parse_axon_program(source), main_module="f")
    typed = typecheck2_flat_axon_file(flat, main_module="f")
    validate_typed_axon_file(typed, main_module="f")
    module = next(module for module in typed.modules if module.name == "f")
    assert isinstance(module.return_type_expr, TypeInt)
    ret_stmt = module.statements[-1]
    assert isinstance(ret_stmt, AxonReturn)
    assert len(ret_stmt.values) == 1
    assert ret_stmt.values[0].inferred_type is not None
    assert isinstance(ret_stmt.values[0].inferred_type, TypeInt)


def test_typecheck_flat_rejects_bare_tensor_type() -> None:
    source = """
main :: Tensor -> Tensor
main x = x
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    with pytest.raises(ValueError, match="Tensor type requires shape dims"):
        typecheck2_flat_axon_file(flat, main_module="main")


def test_typecheck_flat_records_symbolic_module_constraints() -> None:
    source = """
main :: ?Int -> Bool -> Int
main x flag = do
  y <- flag ? 1 : 2
  z <- (x == null) ? 0 : y
  return z
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    assert module.constraints is not None
    assert (
        Constraint(
            relation="=", left="y", right=1, guards=(Constraint(relation="is_true", left="flag"),)
        )
        in module.constraints
    )
    assert (
        Constraint(
            relation="=", left="y", right=2, guards=(Constraint(relation="is_false", left="flag"),)
        )
        in module.constraints
    )
    assert (
        Constraint(
            relation="=", left="z", right=0, guards=(Constraint(relation="is_null", left="x"),)
        )
        in module.constraints
    )


def test_typecheck_threads_guarded_call_constraints_into_callee() -> None:
    source = """
callee :: ?Int -> Int
callee x = (x == null) ? 0 : 1

main :: ?Int -> Int
main x = do
  y <- (x == null) ? (callee x) : 2
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    callee = next(module for module in typed.modules if module.name == "callee")
    assert callee.constraints is not None
    assert any(
        item.relation == "is_null"
        and item.left == "x"
        and len(item.guards) == 1
        and item.guards[0].relation == "callsite"
        and isinstance(item.guards[0].left, str)
        and item.guards[0].left.startswith("main->callee#")
        for item in callee.constraints
    )
    assert any(
        item.relation == "="
        and item.left == "x"
        and item.right == "x"
        and len(item.guards) == 1
        and item.guards[0].relation == "callsite"
        for item in callee.constraints
    )


def test_typecheck_flat_resolves_reshape_shape_alias_name() -> None:
    source = """
main :: Tensor[B,S,H,DH] -> Tensor[B,S,H * DH]
main x = do
  shape <- [B, S, (H * DH)]
  y <- _reshape x shape
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    reshape_line = next(line for line in text.splitlines() if "y <-" in line and "_reshape" in line)
    assert "Tensor[B,S,H * DH]" in reshape_line
    assert "__d" not in reshape_line


def test_typecheck_flat_resolves_expand_shape_alias_name() -> None:
    source = """
main :: Tensor[B,S,D] -> Tensor[B,S,H,DH]
main x = do
  h <- _reshape x [B, S, 1, DH]
  shape <- [B, S, H, DH]
  y <- _expand h shape
  return y
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    expand_line = next(line for line in text.splitlines() if "y <-" in line and "_expand" in line)
    assert "Tensor[B,S,H,DH]" in expand_line
    assert "__d" not in expand_line


def test_typecheck_flat_preserves_cache_update_shape_information() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast
    flat = _flat(resolved, main_module="gpt2")
    typed = typecheck2_flat_axon_file(flat, main_module="gpt2")
    validate_typed_axon_file(typed, main_module="gpt2")
    text = render_axon_file(typed, show_types=True)
    block_sig = next(line for line in text.splitlines() if line.startswith("gpt2_block ::"))
    update_line = next(
        line for line in text.splitlines() if "k, v, new_kv <-" in line and "Cache.update" in line
    )
    assert "Tensor[B,1,S,K]" in block_sig
    assert (
        "?CacheLayer[B,H,P,DH]" in block_sig or "?(Tensor[B,H,P,DH], Tensor[B,H,P,DH])" in block_sig
    )
    assert (
        "?CacheLayer[B,H,K,DH]" in block_sig
        or "?(Tensor[B,H,K,DH], Tensor[B,H,K,DH])" in block_sig
    )
    assert "Any" not in update_line


def test_typecheck_attention_mask_helper_keeps_mask_head_dim_independent(tmp_path: Path) -> None:
    resolved = _resolve_attention_wrapper(tmp_path)
    flat = _flat(resolved, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    mask_sig = next(
        line for line in text.splitlines() if line.startswith("Attention.mask_to_additive ::")
    )
    assert "Tensor[B,H,Q,K] -> Tensor[B,HM,Q,K] -> Tensor[B,H,Q,K]" in mask_sig


def test_typecheck_attention_preserves_matmul_and_mask_broadcast_shapes(tmp_path: Path) -> None:
    resolved = _resolve_attention_wrapper(tmp_path)
    flat = _flat(resolved, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    assert "Tensor[B,H,1,1]" not in text
    scores_line = next(
        line for line in text.splitlines() if "scores <-" in line and "Tensor.matmul" in line
    )
    mask_line = next(
        line for line in text.splitlines() if "mask <-" in line and "Attention.mask_to_additive" in line
    )
    out_line = next(
        line for line in text.splitlines() if "out <-" in line and "Tensor.matmul" in line
    )
    assert "Tensor[B,H,Q,K]" in scores_line
    assert "Tensor[B,H,Q,K]" in mask_line
    assert "Tensor[B,H,Q,VD]" in out_line


def test_typecheck_attention_allows_relative_bias_broadcast(tmp_path: Path) -> None:
    source = """
import Attention (attention)

main :: Tensor[B,H,Q,HD] -> Tensor[B,H,K,HD] -> Tensor[B,H,K,HD] -> Tensor[B,1,Q,K] -> Tensor[B,H,1,K] -> Tensor[B,H,Q,HD]
main q k v keep rel_bias = attention q k v keep rel_bias=rel_bias
"""
    path = tmp_path / "attention_rel_bias_main.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    probs_line = next(
        line
        for line in text.splitlines()
        if "probs_in <-" in line and "rel_bias" in line and "+" in line
    )
    assert "Tensor[B,H,Q,K]" in probs_line


def test_typecheck_transpose_matmul_infers_attention_score_shape(tmp_path: Path) -> None:
    source = """
import Tensor (matmul, transpose)

main :: Tensor[B,H,Q,HD] -> Tensor[B,H,K,HD] -> Tensor[B,H,Q,K]
main q k = do
  kt <- transpose k dim1=2 dim2=3
  scores <- matmul q kt
  return scores
"""
    path = tmp_path / "transpose_matmul.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    kt_line = next(line for line in text.splitlines() if "kt <-" in line)
    scores_line = next(line for line in text.splitlines() if "scores <-" in line)
    assert "Tensor[B,H,HD,K]" in kt_line
    assert "Tensor[B,H,Q,K]" in scores_line


def test_typecheck_rejects_chunk_wrapper_that_claims_unchanged_shape() -> None:
    source = """
bad_chunk :: Tensor[..S] -> List[Tensor[..S]]
bad_chunk x = _chunk x -1 3

main :: Tensor[B,S,D] -> List[Tensor[B,S,D]]
main x = bad_chunk x
"""
    flat = _flat(parse_axon_program(source), main_module="main")
    with pytest.raises(ValueError, match="cannot unify|type mismatch|return"):
        typecheck2_flat_axon_file(flat, main_module="main")


def test_typecheck2_resolves_wrapper_list_args_for_split_type_rule(tmp_path: Path) -> None:
    source = """
import Tensor (split)

main :: Tensor[B,S,D] -> (Tensor[B,S,A], Tensor[B,S,C])
main x = do
  a, c <- split x sizes=[A, C]
  return a, c
"""
    path = tmp_path / "split_main.axon"
    path.write_text(source)
    resolved = resolve_axon_program_from_path(path).ast
    flat = _flat(resolved, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    split_line = next(line for line in text.splitlines() if "a, c <-" in line and "Tensor.split" in line)
    assert "Tensor[B,S,A]" in split_line
    assert "Tensor[B,S,C]" in split_line


def test_typecheck2_gegelu_halves_last_dimension(tmp_path: Path) -> None:
    source = """
import Activations (gegelu)

main :: Tensor[B,S,2 * D] -> Tensor[B,S,D]
main x = gegelu x
"""
    path = tmp_path / "gegelu_main.axon"
    path.write_text(source)
    resolved = resolve_axon_program_from_path(path).ast
    flat = _flat(resolved, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    assert "Activations.gegelu :: Tensor[..S,D] -> ?Float -> Tensor[..S,D / 2]" in text
    assert "main :: Tensor[B,S,2 * D] -> Tensor[B,S,D]" in text
    gegelu_line = next(
        line
        for line in text.splitlines()
        if "Activations.gegelu" in line and "__flat_1 <-" in line
    )
    assert "Tensor[B,S,D]" in gegelu_line


def test_tensor_shape_constructors_typecheck_and_run_with_runtime2(tmp_path: Path) -> None:
    source = """
import Tensor (full, sum, zeros)

main :: Tensor[B,S,D] -> Tensor[B,2,D]
main x = do
  z <- zeros x shape=[B, 2, D]
  y <- full x shape=[B, 2, D] value=1.0
  s <- sum (z + y) dim=1 keepdim=true
  return s + y
"""
    path = tmp_path / "tensor_create.axon"
    path.write_text(source)
    resolved = resolve_axon_program_from_path(path).ast
    flat = _flat(resolved, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    graph = lower_axon_program_to_graph_ir(typed)
    model = make_runtime2_model_class(graph, model_config={}).from_state_dict({})
    out = model.forward(x=torch.randn(3, 5, 7))
    assert tuple(out.shape) == (3, 2, 7)
    assert torch.equal(out, torch.full_like(out, 3.0))


def test_concat_type_rule_updates_concatenated_axis_after_elaborate(tmp_path: Path) -> None:
    source = """
import Tensor (concat)

main :: Tensor[B,S,1] -> Tensor[B,S,1] -> Tensor[B,S,2]
main x y = do
  z <- concat x y dim=-1
  return z
"""
    path = tmp_path / "concat_main.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    bind = next(stmt for stmt in module.statements if isinstance(stmt, AxonBind))
    assert bind.expr.inferred_type == TypeTensor(base="Tensor", dims=("B", "S", 2))


def test_longrope_factors_preserve_explicit_rotary_dim(tmp_path: Path) -> None:
    source = """
import Positions (rope_longrope_factors)

main :: Tensor[B,S] -> (Tensor[B,1,S,96], Tensor[B,1,S,96])
main pos_ids = do
  sin, cos <- rope_longrope_factors pos_ids 96 10000.0 4096 [1.0] [1.0] attention_factor=1.0
  return sin, cos
"""
    path = tmp_path / "longrope_factors.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    assert module.return_type_expr == TypeTuple(
        items=(
            TypeTensor(base="Tensor", dims=("B", 1, "S", 96)),
            TypeTensor(base="Tensor", dims=("B", 1, "S", 96)),
        )
    )


def test_floor_returns_int(tmp_path: Path) -> None:
    source = """
import Math (floor)

main :: Float -> Int
main x = floor x
"""
    path = tmp_path / "floor_returns_int.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    assert module.return_type_expr == TypeInt()


def test_floor_preserves_tensor_shape(tmp_path: Path) -> None:
    source = """
import Math (floor)

main :: Tensor[B,S] -> Tensor[B,S]
main x = floor x
"""
    path = tmp_path / "floor_tensor.axon"
    path.write_text(source)
    flat = _flat(resolve_axon_program_from_path(path).ast, main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    assert module.return_type_expr == TypeTensor(base="Tensor", dims=("B", "S"))


def test_typecheck2_lowers_generic_mamba_without_shape_growth() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/mamba/generic-mamba.axon")
    ).ast
    flat = _flat(resolved, main_module="mamba_2_8b")
    typed = typecheck2_flat_axon_file(flat, main_module="mamba_2_8b")
    validate_typed_axon_file(typed, main_module="mamba_2_8b")
    lower_axon_program_to_graph_ir(typed)
    text = render_axon_file(typed, show_types=True)
    assert "Tensor.concat" not in next(
        module_text
        for module_text in text.split("\n\n")
        if module_text.startswith("SSM.mamba_scan_step ::")
    )


def test_scoped_typevars_does_not_rescope_already_scoped_typevars() -> None:
    ctx = _TcCtx(modules_by_name={}, type_aliases={}, substitutions={}, dim_substitutions={})
    tp = TypeVar(name="Tensor.reshape::Tensor")
    scoped = _scoped_typevars(
        tp, module_name="Tensor.reshape", ctx=ctx, freshen_generics=False
    )
    assert scoped == tp


def test_generic_named_type_rejects_already_scoped_names() -> None:
    assert not _is_generic_named_type(TypeNamed(name="Tensor"), type_aliases={})
    assert not _is_generic_named_type(
        TypeNamed(name="Tensor.reshape::Tensor"),
        type_aliases={},
    )
    assert not _is_generic_named_type(
        TypeNamed(name="TokenIds"),
        type_aliases={"TokenIds": TypeAliasDef(params=(), value=TypeTensor(base="Tensor", dims=()))},
    )
