from __future__ import annotations

from typing import Any

OP_NAME = "mamba_scan"

def compile(emitter, node_spec, env, *, node_path_var, scope_var, indent):
    raise NotImplementedError(f"TinyGrad backend does not yet support op '{OP_NAME}'")

def interpret(model, node_spec, env, *, node_path, scope, symbols):
    raise NotImplementedError(f"TinyGrad backend does not yet support op '{OP_NAME}'")

def uses_node_path(emitter, node_spec):
    return True