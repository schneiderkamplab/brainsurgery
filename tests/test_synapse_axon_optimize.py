from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon.elaborate import elaborate_closed_axon_file
from brainsurgery.synapse.axon.flatten import flatten_closed_axon_file
from brainsurgery.synapse.axon.graph_ir import lower_axon_program_to_graph_ir, validate_graph_program
from brainsurgery.synapse.axon.normalize import normalize_closed_axon_file
from brainsurgery.synapse.axon.optimize import (
    optimize_flat_typed_axon_file,
    optimize_safe_flat_typed_axon_file,
)
from brainsurgery.synapse.axon.parse import parse_axon_program
from brainsurgery.synapse.axon.resolve import resolve_axon_program_from_path
from brainsurgery.synapse.axon.typecheck2 import typecheck2_flat_axon_file
from brainsurgery.synapse.axon.validate import validate_typed_axon_file


def _representative_model_paths() -> tuple[Path, ...]:
    return (
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon"),
        Path("brainsurgery/synapse/models/bert/bert-base-uncased.axon"),
        Path("brainsurgery/synapse/models/llama4/generic-llama4.axon"),
        Path("brainsurgery/synapse/models/olmoe/generic-olmoe.axon"),
    )


def _typecheck_model_path(axon_path: Path):
    resolved = resolve_axon_program_from_path(axon_path).ast
    normalized = normalize_closed_axon_file(resolved)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    return typecheck2_flat_axon_file(flat)


def test_optimize_safe_prunes_unreachable_and_folds_literals() -> None:
    source = """
unused :: Int
unused = 2

main :: Int
main = do
  x <- 1 + 2
  y <- x
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_safe_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    text = "\n".join(str(stmt) for module in optimized.modules for stmt in module.statements)
    assert [module.name for module in optimized.modules] == ["main"]
    assert "value=3" in text


def test_optimize_safe_promotes_generated_return_alias_to_public_name() -> None:
    source = """
main :: Bool -> (Int, Int)
main flag = do
  __flat_1 <- flag ? 1 : 2
  logits <- __flat_1
  return logits, 0
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_safe_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "targets=('logits',)" in text
    assert "targets=('__flat_1',)" not in text
    graph = lower_axon_program_to_graph_ir(optimized, main_module="main")
    graph_main = next(module for module in graph.modules if module.name == graph.main_module)
    assert graph_main.outputs[0].name == "logits"


@pytest.mark.parametrize("axon_path", _representative_model_paths(), ids=lambda path: path.as_posix())
def test_optimize_safe_representative_models_validate_and_lower(axon_path: Path) -> None:
    typed = _typecheck_model_path(axon_path)
    optimized = optimize_safe_flat_typed_axon_file(typed)
    validate_typed_axon_file(optimized)
    graph = lower_axon_program_to_graph_ir(optimized)
    validate_graph_program(graph)


def test_optimize_safe_does_not_delete_unused_calls() -> None:
    source = """
helper :: Int -> Int
helper x = do
  y <- x + 1
  return y

main :: Int -> Int
main x = do
  unused <- helper x
  return x
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_safe_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    assert any("helper" in str(stmt) for stmt in main.statements)


def test_optimize_inlines_alias_definition_calls() -> None:
    source = """
id :: Int -> Int
id x = x

main :: Int -> Int
main x = do
  y <- id x
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    assert all(module.name != "id" for module in optimized.modules)
    assert len(main.statements) == 1
    assert "id" not in str(main.statements[0])


def test_optimize_folds_constant_ternary_and_alias_binds() -> None:
    source = """
main :: Int
main = do
  x <- true ? 1 : 2
  y <- x
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    assert len(main.statements) == 2
    assert "2" not in str(main.statements[0])
    assert "true" not in str(main.statements[0]).lower()


def test_optimize_does_not_specialize_shared_callee_through_unresolved_forwarder() -> None:
    source = """
wrap :: Float -> Float
wrap scale = do
  y <- scale + 1.0
  return y

helper :: Float -> Float
helper embed_scale = wrap embed_scale

main :: Float
main = do
  a <- wrap 39.0
  b <- helper 16.0
  c <- a + b
  return c
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    text = "\n".join(str(stmt) for module in optimized.modules for stmt in module.statements)
    assert "57.0" in text
    assert "80.0" not in text


def test_optimize_specializes_single_entry_recursive_helper_params() -> None:
    source = """
loop_continue :: Int -> Int -> Int -> Int -> Int
loop_continue i limit delta acc = loop_recur (i + delta) limit delta acc

loop_recur :: Int -> Int -> Int -> Int -> Int
loop_recur i limit delta acc = do
  positive <- delta > 0
  ge_limit <- i >= limit
  le_limit <- i <= limit
  stop <- positive ? ge_limit : le_limit
  result <- stop ? acc : loop_continue i limit delta acc
  return result

main :: Int -> Int
main acc = loop_recur 0 10 1 acc
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    recur = next(module for module in optimized.modules if module.name == "loop_recur")
    cont = next(module for module in optimized.modules if module.name == "loop_continue")
    assert [param.name for param in recur.params] == ["i", "acc"]
    assert [param.name for param in cont.params] == ["i", "acc"]
    text = "\n".join(str(stmt) for stmt in recur.statements)
    assert "value=0), right=AxonExprName" not in text
    assert "name='i'), right=AxonExprInt" in text
    assert "name='le_limit'" not in text
    assert "name='positive'" not in text


def test_optimize_does_not_specialize_rebound_recursive_carry_to_null() -> None:
    source = """
loop_continue :: Int -> Int -> Int
loop_continue i state = do
  state <- (i == 0) ? 1 : state
  next <- i + 1
  out <- loop_recur next state
  return out

loop_recur :: Int -> Int -> Int
loop_recur i state = do
  flag <- i >= 3
  out <- flag ? state : (loop_continue i state)
  return out

main :: Int
main = loop_recur 0 0
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    recur = next(module for module in optimized.modules if module.name == "loop_recur")
    cont = next(module for module in optimized.modules if module.name == "loop_continue")
    assert "state" in [param.name for param in recur.params]
    assert "state" in [param.name for param in cont.params]


def test_optimize_prunes_unused_path_and_value_params() -> None:
    source = """
helper :: Path -> Int -> Int -> Int
helper@scope used unused = do
  return used

main :: Int -> Int
main x = helper @@w x 7
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    assert all(module.name != "helper" for module in optimized.modules)
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "@@w" not in text
    assert "7" not in text


def test_optimize_repeatedly_prunes_transitively_unused_scope_params() -> None:
    source = """
inner :: Path -> Int -> Int
inner@scope x = do
  return x

outer :: Path -> Int -> Int
outer@scope x = do
  y <- inner scope x
  return y

main :: Int -> Int
main x = outer @@w x
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    assert all(module.name != "inner" for module in optimized.modules)
    assert all(module.name != "outer" for module in optimized.modules)


def test_optimize_folds_boolean_identities() -> None:
    source = """
main :: Bool -> Bool
main flag = do
  x <- true or flag
  y <- x and true
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "or" not in text
    assert "and" not in text
    assert "flag" not in text


def test_optimize_inlines_single_callsite_multistatement_module_with_templated_path() -> None:
    source = """
helper :: Path -> Tensor[B,S,D] -> Tensor[B,S,D]
helper scope x = do
  y <- _layernorm scope x 1e-5 null @@w true @@b
  return y

main :: Int -> Tensor[B,S,D] -> Tensor[B,S,D]
main i x = do
  y <- helper @@'h.{i}' x
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    assert all(module.name != "helper" for module in optimized.modules)
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "AxonExprPath" in text
    assert "{i}" in text


def test_optimize_inlines_expression_position_helper_and_prunes_definition() -> None:
    source = """
helper :: Int -> Int -> Int
helper x y = do
  z <- x + 1
  return z + y

main :: Bool -> Int -> Int
main flag x = do
  y <- flag ? x : helper x 2
  return y
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "flag" in text


def test_optimize_folds_ternaries_from_constraints_for_reused_guard() -> None:
    source = """
main :: Int
main = do
  guard <- true
  a <- guard ? 1 : 2
  b <- guard ? 3 : 4
  return a
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert " ? " not in text
    assert "2" not in text
    assert "4" not in text


def test_optimize_propagates_local_bool_facts_for_ternary_folding() -> None:
    source = """
main :: Int
main = do
  guard <- 1 == 1
  x <- guard ? 1 : 2
  return x
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert " ? " not in text
    assert "2" not in text


def test_optimize_inlines_single_use_pure_non_atomic_bind_when_flat_shape_survives() -> None:
    source = """
main :: Bool -> Bool -> Int -> Int
main flag keep x = do
  y <- flag ? x : 2
  z <- keep ? 0 : y
  return z
"""
    flat = flatten_closed_axon_file(parse_axon_program(source), main_module="main")
    typed = typecheck2_flat_axon_file(flat, main_module="main")
    optimized = optimize_flat_typed_axon_file(typed, main_module="main")
    validate_typed_axon_file(optimized, main_module="main")
    main = next(module for module in optimized.modules if module.name == "main")
    text = "\n".join(str(stmt) for stmt in main.statements)
    assert "targets=('y',)" not in text


def test_optimize_rewrites_list_destructuring_to_index_binds() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/gpt2.axon")
    ).ast
    normalized = normalize_closed_axon_file(resolved, main_module="gpt2")
    elaborated = elaborate_closed_axon_file(normalized, main_module="gpt2")
    flat = flatten_closed_axon_file(elaborated, main_module="gpt2")
    typed = typecheck2_flat_axon_file(flat, main_module="gpt2")
    optimized = optimize_flat_typed_axon_file(typed, main_module="gpt2")
    validate_typed_axon_file(optimized, main_module="gpt2")
    text = "\n".join(str(stmt) for module in optimized.modules for stmt in module.statements)
    assert "callee='_list_index'" in text


def test_optimize_reapplies_structural_passes_until_fixpoint_on_generic_gpt2() -> None:
    resolved = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast
    normalized = normalize_closed_axon_file(resolved, main_module="gpt2")
    elaborated = elaborate_closed_axon_file(normalized, main_module="gpt2")
    flat = flatten_closed_axon_file(elaborated, main_module="gpt2")
    typed = typecheck2_flat_axon_file(flat, main_module="gpt2")
    optimized = optimize_flat_typed_axon_file(typed, main_module="gpt2")
    validate_typed_axon_file(optimized, main_module="gpt2")
    module_names = {module.name for module in optimized.modules}
    assert "gpt2" in module_names
    assert any(name.startswith("gpt2__loop_h_recur_") for name in module_names)
    assert not any(name.startswith("gpt2__loop_h_step_") for name in module_names)
