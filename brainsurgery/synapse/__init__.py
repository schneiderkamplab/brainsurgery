from .axon import (
    AxonBind,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    MaterializeContext,
    ResolveDiagnostic,
    ResolveReport,
    ast_equal,
    checkpoint_pragma_entries,
    flatten_closed_axon_file,
    group_output_name,
    load_materialize_context,
    load_tokenizer,
    lower_axon_module_to_synapse_block,
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    materialize_axon_file,
    normalize_checkpoint_name,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
    render_axon_file,
    resolve_axon_program_from_path,
    resolve_axon_program_to_source,
    resolve_loaded_axon_files,
    synapse_spec_to_axon_module_text,
    validate_axon_program,
    validate_closed_axon_file,
    validate_flat_axon_file,
)
from .codegen import emit_model_code_from_synapse_spec, load_synapse_torch_op_map
from .pipeline_backend import (
    PipelinePlan,
    PipelineStage,
    available_pipeline_devices,
    build_pipeline_plan,
    build_pipeline_stage_spec,
    build_pipeline_stage_specs,
    partition_layer_ranges,
)
from .pipeline_codegen import (
    emit_pipeline_stage_code_from_synapse_spec,
    emit_pipeline_stage_codes_from_synapse_spec,
)
from .pipeline_runtime import SynapsePipelineModel, build_hf_device_map_from_pipeline_usage
from .runtime import SynapseProgramModel
from .type_inference import (
    annotate_spec_with_block_io_types,
    extract_block_io_types_from_spec,
    infer_block_io_types_from_modules,
    infer_output_types_for_node,
)
from .visualize import (
    render_synapse_spec_to_dot,
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


def run_axon_op_parity(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .op_parity import run_axon_op_parity as _run_axon_op_parity

    return _run_axon_op_parity(*args, **kwargs)


def run_axon_layer_op_parity(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .op_parity import run_axon_layer_op_parity as _run_axon_layer_op_parity

    return _run_axon_layer_op_parity(*args, **kwargs)


def run_codegen_runtime_parity(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .op_parity import run_codegen_runtime_parity as _run_codegen_runtime_parity

    return _run_codegen_runtime_parity(*args, **kwargs)


def run_axon_visualize(*args, **kwargs):
    from .visualize import run_axon_visualize as _run_axon_visualize

    return _run_axon_visualize(*args, **kwargs)


def render_axon_benchmark_log(*args, **kwargs):
    from .axon_test_log import render_axon_benchmark_log as _render_axon_benchmark_log

    return _render_axon_benchmark_log(*args, **kwargs)


def render_axon_test_log(*args, **kwargs):
    from .axon_test_log import render_axon_test_log as _render_axon_test_log

    return _render_axon_test_log(*args, **kwargs)


__all__ = [
    "AxonBind",
    "AxonModule",
    "AxonParam",
    "AxonRepeat",
    "AxonReturn",
    "MaterializeContext",
    "SynapseProgramModel",
    "ast_equal",
    "annotate_spec_with_block_io_types",
    "checkpoint_pragma_entries",
    "emit_model_code_from_synapse_spec",
    "extract_block_io_types_from_spec",
    "group_output_name",
    "flatten_closed_axon_file",
    "infer_block_io_types_from_modules",
    "infer_output_types_for_node",
    "PipelinePlan",
    "PipelineStage",
    "SynapsePipelineModel",
    "build_hf_device_map_from_pipeline_usage",
    "available_pipeline_devices",
    "build_pipeline_plan",
    "build_pipeline_stage_spec",
    "build_pipeline_stage_specs",
    "emit_pipeline_stage_code_from_synapse_spec",
    "emit_pipeline_stage_codes_from_synapse_spec",
    "load_materialize_context",
    "partition_layer_ranges",
    "run_axon_benchmark",
    "render_axon_benchmark_csv",
    "run_axon_test",
    "render_axon_benchmark_log",
    "render_axon_test_log",
    "run_axon_test_matrix",
    "run_axon_op_parity",
    "run_axon_layer_op_parity",
    "run_codegen_runtime_parity",
    "run_axon_visualize",
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
    "load_tokenizer",
    "load_synapse_torch_op_map",
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
    "materialize_axon_file",
    "normalize_checkpoint_name",
    "render_synapse_spec_to_dot",
    "render_axon_file",
    "resolve_loaded_axon_files",
    "resolve_axon_program_from_path",
    "resolve_axon_program_to_source",
    "ResolveDiagnostic",
    "ResolveReport",
    "synapse_spec_to_axon_module_text",
    "validate_closed_axon_file",
    "validate_flat_axon_file",
    "validate_axon_program",
]
