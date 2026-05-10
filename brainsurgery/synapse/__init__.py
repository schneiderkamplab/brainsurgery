from .axon import (
    AxonBind,
    AxonDefinition,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    MaterializeContext,
    ResolveDiagnostic,
    ResolveReport,
    ast_equal,
    canonicalize_typed_axon_file,
    checkpoint_pragma_entries,
    flatten_closed_axon_file,
    group_output_name,
    graph_program_to_axon_file,
    load_materialize_context,
    load_tokenizer,
    lower_axon_program_to_graph_ir,
    materialize_axon_file,
    normalize_checkpoint_name,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
    render_axon_file,
    resolve_axon_program_from_path,
    resolve_axon_program_to_source,
    resolve_loaded_axon_files,
    validate_axon_program,
    validate_closed_axon_file,
    validate_flat_axon_file,
)
from .type_inference import (
    annotate_spec_with_block_io_types,
    extract_block_io_types_from_spec,
    infer_block_io_types_from_modules,
    infer_output_types_for_node,
)
def run_axon_test(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_test import run_axon_test as _run_axon_test

    return _run_axon_test(*args, **kwargs)


def run_axon_benchmark(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_benchmark import run_axon_benchmark as _run_axon_benchmark

    return _run_axon_benchmark(*args, **kwargs)


def render_axon_benchmark_csv(*args, **kwargs):
    from .axon_benchmark import render_axon_benchmark_csv as _render_axon_benchmark_csv

    return _render_axon_benchmark_csv(*args, **kwargs)


def run_axon_test_matrix(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_test_matrix import run_axon_test_matrix as _run_axon_test_matrix

    return _run_axon_test_matrix(*args, **kwargs)


def render_axon_benchmark_log(*args, **kwargs):
    from .axon_test_log import render_axon_benchmark_log as _render_axon_benchmark_log

    return _render_axon_benchmark_log(*args, **kwargs)


def render_axon_test_log(*args, **kwargs):
    from .axon_test_log import render_axon_test_log as _render_axon_test_log

    return _render_axon_test_log(*args, **kwargs)


__all__ = [
    "AxonBind",
    "AxonDefinition",
    "AxonParam",
    "AxonRepeat",
    "AxonReturn",
    "MaterializeContext",
    "ast_equal",
    "annotate_spec_with_block_io_types",
    "checkpoint_pragma_entries",
    "extract_block_io_types_from_spec",
    "group_output_name",
    "graph_program_to_axon_file",
    "flatten_closed_axon_file",
    "canonicalize_typed_axon_file",
    "infer_block_io_types_from_modules",
    "infer_output_types_for_node",
    "load_materialize_context",
    "run_axon_benchmark",
    "render_axon_benchmark_csv",
    "run_axon_test",
    "render_axon_benchmark_log",
    "render_axon_test_log",
    "run_axon_test_matrix",
    "lower_axon_program_to_graph_ir",
    "load_tokenizer",
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
    "materialize_axon_file",
    "normalize_checkpoint_name",
    "render_axon_file",
    "resolve_loaded_axon_files",
    "resolve_axon_program_from_path",
    "resolve_axon_program_to_source",
    "ResolveDiagnostic",
    "ResolveReport",
    "validate_closed_axon_file",
    "validate_flat_axon_file",
    "validate_axon_program",
]
