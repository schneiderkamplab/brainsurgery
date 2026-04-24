from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse.axon.ast import (
    AxonBind,
    AxonRepeat,
    AxonReturn,
    Constraint,
    TypeAliasDef,
    TypeDim,
    TypeInt,
    TypeNamed,
    TypeTensor,
    TypeVar,
)
from brainsurgery.synapse.axon.ast.render import render_axon_file
from brainsurgery.synapse.axon.flatten import flatten_closed_axon_file
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.typecheck.core import _TcCtx, _is_generic_named_type, _scoped_typevars
from brainsurgery.synapse.axon.typecheck import typecheck_flat_axon_file
from brainsurgery.synapse.axon.validate import validate_typed_axon_file


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
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    helper = next(
        module
        for module in typed.modules
        if module.name.startswith("main__loop_h_recur_") and "_recur_continue_" not in module.name
    )
    assert isinstance(helper.params[0].type_expr, TypeInt)
    assert isinstance(helper.params[3].type_expr, TypeTensor)
    assert isinstance(helper.return_type_expr, TypeTensor)


def test_typecheck_flat_populates_expr_annotations() -> None:
    source = """
f :: Int -> Int
f x = x

main :: Int -> Int
main x = do
  y <- f x
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    module = next(module for module in typed.modules if module.name == "main")
    bind_stmt = next(stmt for stmt in module.statements if isinstance(stmt, AxonBind))
    assert bind_stmt.expr.inferred_type is not None
    assert bind_stmt.expr.inferred_arity == 1
    repeat_stmts = [stmt for stmt in module.statements if isinstance(stmt, AxonRepeat)]
    assert repeat_stmts == []


def test_typecheck_flat_unifies_embedding_dim_from_add() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast
    flat = flatten_closed_axon_file(resolved, main_module="gpt2")
    typed = typecheck_flat_axon_file(flat, main_module="gpt2")
    validate_typed_axon_file(typed, main_module="gpt2")
    text = render_axon_file(typed, show_types=True)
    pos_line = next(
        line for line in text.splitlines() if "pos <- (NN.embedding (@@wpe :: Path)" in line
    )
    layernorm_line = next(line for line in text.splitlines() if "__flat_3 <- (NN.layernorm" in line)
    assert "MODEL_DIM" in pos_line
    assert ",dim]" not in pos_line
    assert "Tensor[" in layernorm_line and ",MODEL_DIM]" in layernorm_line


def test_typecheck_flat_exposes_signature_dim_as_term_dim() -> None:
    source = """
f :: Tensor[B,S,D] -> Int
f x = D
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="f")
    typed = typecheck_flat_axon_file(flat, main_module="f")
    validate_typed_axon_file(typed, main_module="f")
    module = next(module for module in typed.modules if module.name == "f")
    assert isinstance(module.return_type_expr, TypeInt)
    ret_stmt = module.statements[-1]
    assert isinstance(ret_stmt, AxonReturn)
    assert len(ret_stmt.values) == 1
    assert ret_stmt.values[0].inferred_type is not None
    assert isinstance(ret_stmt.values[0].inferred_type, TypeDim)


def test_typecheck_flat_records_symbolic_module_constraints() -> None:
    source = """
main :: ?Int -> Bool -> Int
main x flag = do
  y <- flag ? 1 : 2
  z <- (x == null) ? 0 : y
  return z
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
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
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
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
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    reshape_line = next(line for line in text.splitlines() if "y <- (_reshape" in line)
    assert "Tensor[B,S,H * DH]" in reshape_line
    assert "__d" not in reshape_line


def test_typecheck_flat_preserves_cache_update_shape_information() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast
    flat = flatten_closed_axon_file(resolved, main_module="gpt2")
    typed = typecheck_flat_axon_file(flat, main_module="gpt2")
    validate_typed_axon_file(typed, main_module="gpt2")
    text = render_axon_file(typed, show_types=True)
    block_sig = next(line for line in text.splitlines() if line.startswith("gpt2_block ::"))
    update_line = next(
        line for line in text.splitlines() if "k, v, new_kv <- (Cache.update " in line
    )
    assert "?Tensor[B,K]" in block_sig
    assert (
        "?CacheLayer[B,H,P,DH]" in block_sig or "?(Tensor[B,H,P,DH], Tensor[B,H,P,DH])" in block_sig
    )
    assert (
        "?CacheLayer[B,H,K,DH]" in block_sig or "?(Tensor[B,H,K,DH], Tensor[B,H,K,DH])" in block_sig
    )
    assert "Any" not in update_line


def test_typecheck_attention_mask_helper_does_not_force_equal_shapes() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/builtins/Attention.axon")
    ).ast
    flat = flatten_closed_axon_file(resolved, main_module="attention")
    typed = typecheck_flat_axon_file(flat, main_module="attention")
    validate_typed_axon_file(typed, main_module="attention")
    text = render_axon_file(typed, show_types=True)
    mask_sig = next(line for line in text.splitlines() if line.startswith("mask_to_additive ::"))
    assert "Tensor[..S] -> Tensor[..M] -> Tensor[..S]" in mask_sig


def test_typecheck_attention_preserves_matmul_and_mask_broadcast_shapes() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/builtins/Attention.axon")
    ).ast
    flat = flatten_closed_axon_file(resolved, main_module="attention")
    typed = typecheck_flat_axon_file(flat, main_module="attention")
    validate_typed_axon_file(typed, main_module="attention")
    text = render_axon_file(typed, show_types=True)
    assert "Tensor[B,H,1,1]" not in text
    assert "Tensor[B,H,Q,K]" in next(
        line for line in text.splitlines() if "scores <- (Tensor.matmul" in line
    )
    assert "Tensor[B,H,Q,K]" in next(
        line for line in text.splitlines() if "mask <- (mask_to_additive" in line
    )
    assert "Tensor[B,H,Q,HD]" in next(
        line for line in text.splitlines() if "out <- (Tensor.matmul" in line
    )


def test_typecheck_chunk_stays_list_typed_with_constant_parts() -> None:
    source = """
main :: Tensor[B,S,D] -> List[Tensor[B,S,D]]
main x = do
  y <- _chunk x -1 3
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck_flat_axon_file(flat, main_module="main")
    validate_typed_axon_file(typed, main_module="main")
    text = render_axon_file(typed, show_types=True)
    assert "_chunk" in text
    assert "List[Tensor[B,S,D]]" in text


def test_scoped_typevars_does_not_rescope_already_scoped_typevars() -> None:
    ctx = _TcCtx(modules_by_name={}, type_aliases={}, substitutions={}, dim_substitutions={})
    tp = TypeVar(name="Tensor.reshape::Tensor")
    scoped = _scoped_typevars(
        tp, module_name="Tensor.reshape", ctx=ctx, freshen_generics=False
    )
    assert scoped == tp


def test_generic_named_type_rejects_already_scoped_names() -> None:
    assert _is_generic_named_type(TypeNamed(name="Tensor"), type_aliases={})
    assert not _is_generic_named_type(
        TypeNamed(name="Tensor.reshape::Tensor"),
        type_aliases={},
    )
    assert not _is_generic_named_type(
        TypeNamed(name="TokenIds"),
        type_aliases={"TokenIds": TypeAliasDef(params=(), value=TypeTensor(base="Tensor", dims=()))},
    )
