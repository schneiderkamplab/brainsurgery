from .axon import (
    TYPING_RULES,
    AxonBind,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    load_tokenizer,
    lower_axon_module_to_synapse_block,
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
    synapse_spec_to_axon_module_text,
    typecheck_axon_module,
    typecheck_axon_program,
    validate_axon_program,
)
from .codegen import emit_model_code_from_synapse_spec, load_synapse_torch_op_map
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


def run_axon_test_matrix(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_test_matrix import run_axon_test_matrix as _run_axon_test_matrix

    return _run_axon_test_matrix(*args, **kwargs)


def run_axon_op_parity(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .op_parity import run_axon_op_parity as _run_axon_op_parity

    return _run_axon_op_parity(*args, **kwargs)


def run_codegen_runtime_parity(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .op_parity import run_codegen_runtime_parity as _run_codegen_runtime_parity

    return _run_codegen_runtime_parity(*args, **kwargs)


def run_axon_visualize(*args, **kwargs):
    from .visualize import run_axon_visualize as _run_axon_visualize

    return _run_axon_visualize(*args, **kwargs)


__all__ = [
    "AxonBind",
    "AxonModule",
    "AxonParam",
    "AxonRepeat",
    "AxonReturn",
    "SynapseProgramModel",
    "annotate_spec_with_block_io_types",
    "emit_model_code_from_synapse_spec",
    "extract_block_io_types_from_spec",
    "infer_block_io_types_from_modules",
    "infer_output_types_for_node",
    "run_axon_test",
    "run_axon_test_matrix",
    "run_axon_op_parity",
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
    "render_synapse_spec_to_dot",
    "synapse_spec_to_axon_module_text",
    "typecheck_axon_module",
    "typecheck_axon_program",
    "TYPING_RULES",
    "validate_axon_program",
]
