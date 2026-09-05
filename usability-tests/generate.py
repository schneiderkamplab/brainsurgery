#!/usr/bin/env python
"""Render everything that depends on the surgery target from targets.py.

    .venv/bin/python usability-tests/generate.py

Writes, for every target in targets.TARGETS and every test:

    tasks/<test>/TASK-<target>.md          participant-facing specification
    solutions/<target>/P/T<n>.py           reference Python baseline (hidden)
    solutions/<target>/P/_ckpt.py            shared loader / sharded writer for the baselines
    solutions/<target>/B/T<n>.yaml         reference BrainSurgery plan (hidden)
    review/<target>/P/T<n>-defective.py    baseline with one injected defect (bug-detection phase)
    review/<target>/B/T<n>-defective.yaml  plan with the same injected defect
    review/<target>/answers.json           what the defect is, per test

Re-run after editing targets.py. Hand edits to generated files are lost.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

from targets import TARGETS, TESTS, esc, head_keep_slices, kept_layers, pruned_size

HERE = Path(__file__).resolve().parent


def render(template: str, **kw) -> str:
    return Template(template).substitute(**kw)


def base_ref(t: dict) -> str:
    return "inputs/base" if t["base_layout"] == "sharded" else "inputs/base/model.safetensors"


def base_desc(t: dict) -> str:
    if t["base_layout"] == "sharded":
        return ("`inputs/base/`: " + t["display"] + " as a sharded safetensors directory "
                "(two shard files plus `model.safetensors.index.json`)")
    return f"`inputs/base/model.safetensors`: {t['display']}"


def slice_expr(dim: int, lo: int, hi: int) -> str:
    return f"[{lo}:{hi}]" if dim == 0 else f"[:, {lo}:{hi}]"


# --------------------------------------------------------------------------- shared

IO_PY = '''"""Shared helpers for the Python baselines: load a file or sharded directory, write shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_file():
        return load_file(str(path))
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(str(path / shard)))
        return sd
    return load_file(str(path / "model.safetensors"))


def save_sharded_safetensors(sd: dict[str, torch.Tensor], out_dir: Path, max_bytes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_size = 0
    for name, tensor in sd.items():
        size = tensor.numel() * tensor.element_size()
        if cur and cur_size + size > max_bytes:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = tensor.contiguous()
        cur_size += size
    if cur:
        shards.append(cur)
    weight_map: dict[str, str] = {}
    for idx, shard in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / shard_name))
        for name in shard:
            weight_map[name] = shard_name
    total = sum(t.numel() * t.element_size() for t in sd.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
'''

ENV_MD = """## Environment

This task runs in its own sandbox: a fresh working directory and a fresh
Python environment that contains only the packages of your condition. Nothing
from other tasks, other conditions or earlier runs is available, and nothing
you do here is visible to them. Inputs are under `inputs/` (read-only). Write
only under `out/`. Do not leave the sandbox directory.
"""


# --------------------------------------------------------------------------- T1

T1_TASK = """# T1: Depth pruning with layer renumbering (${display})

## Objective

Produce a ${n_kept}-layer ${display} checkpoint by removing ${n_drop} transformer
blocks from the ${n_layers}-layer model and renumbering the remaining blocks so
that layer indices are contiguous again. This is the checkpoint side of depth
pruning: the result must load into a ${n_kept}-layer configuration of the same
architecture.

## Why it is meaningful

Depth pruning is a standard way to shrink a model for cheaper inference or to
build a student for distillation: drop whole transformer blocks, then keep
training. The checkpoint side is a bulk rename with a collision hazard: if
blocks are shifted in the wrong order, a block overwrites a surviving one, and
the result still loads and runs, silently wrong. A correct solution has to
target whole blocks by pattern, delete them, renumber the rest without
collisions, and prove afterwards that exactly ${n_kept} blocks remain.

${env}
## Input

- ${base_desc}: ${total} tensors, ${base_dtype}.
  Each transformer block `i` in 0..${last_layer} owns ${tpl} tensors named
  `${layer_example}<rest>`: ${layer_tensors_note}. The
  remaining ${n_nonlayer} tensors are ${nonlayer_list}. ${layout_note} The
  directory also holds the HuggingFace config and tokenizer files of the
  ${n_layers}-layer model.

## Required result

1. Remove every tensor of blocks ${drop_list}.
2. Renumber the surviving blocks in their original order so that indices run
   0..${last_kept} without gaps: ${renumber_text}. Only the block index in the
   name changes; the rest of each name, and all values, shapes and dtypes,
   stay the same.
3. The ${n_nonlayer} non-block tensors are unchanged.
4. Output: a single file `out/T1/model.safetensors` with exactly ${t1_total}
   tensors.

## Required checks

Your solution must fail loudly (non-zero exit, no output written) if any of
these does not hold:

- no tensor of blocks ${gone_list} remains;
- exactly ${n_kept} blocks remain (for example, exactly ${n_kept} tensors match
  `${layer_example}${probe_rel}`);
- the output has exactly ${t1_total} tensors.

## Grading

`grade.py T1 --target ${target}` compares `out/T1` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
"""

T1_PY = '''"""T1 baseline for ${display}: remove blocks ${drop_list} and renumber the rest contiguously."""

import re
import sys
from pathlib import Path

from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T1")
DROP = {${drop_set}}
N_LAYERS = ${n_layers}

sd = load_checkpoint("${base_ref}")
layer_re = re.compile(r"${layer_re}(.+)")
kept = [i for i in range(N_LAYERS) if i not in DROP]
renumber = {old: new for new, old in enumerate(kept)}

out = {}
for name, tensor in sd.items():
    match = layer_re.fullmatch(name)
    if match is None:
        out[name] = tensor
        continue
    old = int(match.group(1))
    if old in DROP:
        continue
    out["${layer_fmt}".format(i=renumber[old]) + match.group(2)] = tensor

layers = {int(m.group(1)) for n in out if (m := layer_re.fullmatch(n))}
assert layers == set(range(len(kept))), sorted(layers)
assert len(out) == ${t1_total}, len(out)

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
'''

T1_YAML = """# T1 for ${display}: remove blocks ${drop_list} and renumber the rest contiguously.
inputs:
  - model::${base_ref}
transforms:
  - assert: { count: { of: '${drop_glob}', is: ${n_drop_tensors} } }
  - delete: { target: '${drop_glob}' }
  # Shift surviving blocks down, lowest first, so each destination is free.
${moves}
  - assert: { not: { exists: '${gone_glob}' } }
  - assert: { count: { of: '${probe_glob}', is: ${n_kept} } }
  - assert: { count: { of: '.*', is: ${t1_total} } }
output: out/T1/model.safetensors
"""


def gen_t1(name: str, t: dict, drop: list[int]) -> tuple[str, str, str]:
    kept = [i for i in range(t["n_layers"]) if i not in drop]
    fmt = t["layer_fmt"]
    n_drop_t = len(drop) * t["tensors_per_layer"]
    t1_total = t["total_tensors"] - n_drop_t
    gone = list(range(len(kept), t["n_layers"]))
    renumber_pairs = [(old, new) for new, old in enumerate(kept)]
    renumber_text = ", ".join(
        f"old {o} stays {o}" if o == n else f"old {o} becomes {n}" for o, n in renumber_pairs
    )
    probe_rel = t["proj_matrices"][0][0]
    task = render(
        T1_TASK, display=t["display"], n_kept=len(kept), n_drop=len(drop), n_layers=t["n_layers"],
        env=ENV_MD, base_desc=base_desc(t), total=t["total_tensors"], base_dtype=t["base_dtype"],
        last_layer=t["n_layers"] - 1, tpl=t["tensors_per_layer"], layer_example=fmt.format(i="<i>"),
        layer_tensors_note=t["layer_tensors_note"], n_nonlayer=len(t["nonlayer"]),
        nonlayer_list=", ".join(f"`{n}` (`{s}`)" for n, s in t["nonlayer"]), layout_note=t["layout_note"],
        drop_list=", ".join(map(str, drop)), last_kept=len(kept) - 1, renumber_text=renumber_text,
        t1_total=t1_total, gone_list=", ".join(map(str, gone)), probe_rel=probe_rel, target=name,
    )
    py = render(
        T1_PY, display=t["display"], drop_list=", ".join(map(str, drop)), drop_set=", ".join(map(str, drop)),
        n_layers=t["n_layers"], base_ref=base_ref(t), layer_re=t["layer_re"], layer_fmt=fmt, t1_total=t1_total,
    )
    glob = t["layer_glob"]
    moves = "\n".join(
        f"  - move: {{ from: '{glob.format(i=o)}(.*)', to: '{fmt.format(i=n)}\\1' }}"
        for o, n in renumber_pairs if o != n
    )
    yaml = render(
        T1_YAML, display=t["display"], drop_list=", ".join(map(str, drop)), base_ref=base_ref(t),
        drop_glob=glob.format(i="(" + "|".join(map(str, drop)) + ")") + ".*", n_drop_tensors=n_drop_t,
        moves=moves, gone_glob=glob.format(i="(" + "|".join(map(str, gone)) + ")") + ".*",
        probe_glob=t["layer_re"] + esc(probe_rel), n_kept=len(kept), t1_total=t1_total,
    )
    return task, py, yaml


# --------------------------------------------------------------------------- T2

T2_TASK = """# T2: Structured attention-head pruning (${display})

## Objective

Remove one attention head from every layer of ${display} at the checkpoint
level. Pruning a head means removing its slice from every head-bearing
projection tensor: the input-side projections that produce the head's query,
key and value, and the output projection that consumes it. The result must be
loadable as the same architecture with ${n_heads_kept} heads per layer.

## Why it is meaningful

Structured head pruning (removing whole attention heads found to be
redundant) is a well-studied way to speed up transformers, and it is done at
the checkpoint level so the pruned model loads with a smaller head count. The
work is in the layout: which axis of which tensor holds the heads, whether
query, key and value are fused, and in what order. Getting a block boundary
wrong produces a checkpoint that loads and runs with garbage attention. A
correct solution has to slice and reassemble tensors, keep the piece order
right, and check the resulting shapes on every projection.

${env}
## Input

- ${base_desc}: ${total} tensors, ${base_dtype}, ${n_layers} layers,
  ${n_heads} heads of ${head_dim} dimensions each, hidden size ${hidden}.
  ${layout_note}

Per layer `i` in 0..${last_layer} the head-bearing tensors are:

${head_tensor_list}

Not per head, and therefore untouched: ${untouched}.

## Required result

For every layer `i`, remove head ${drop_head} (heads are numbered from 0):

${result_items}
${next_item}. Every other tensor is unchanged. Tensor names do not change.
${out_item}. Output: a single file `out/T2/model.safetensors` with exactly ${total} tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

${check_items}
- the output has exactly ${total} tensors.

## Grading

`grade.py T2 --target ${target}` compares `out/T2` with a hidden reference:
exact key set, shapes, dtypes and bit-exact values.
"""

T2_PY = '''"""T2 baseline for ${display}: remove attention head ${drop_head} from every layer."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T2")
N_LAYERS = ${n_layers}
DROP_HEAD = ${drop_head}
# name, axis holding heads, number of concatenated segments, segment width, head block width
SPECS = ${specs}


def keep_index(segments: int, seg_size: int, block: int) -> torch.Tensor:
    keep = []
    for seg in range(segments):
        for h in range(seg_size // block):
            if h != DROP_HEAD:
                start = seg * seg_size + h * block
                keep.append(torch.arange(start, start + block))
    return torch.cat(keep)


sd = load_checkpoint("${base_ref}")
for layer in range(N_LAYERS):
    for rel, dim, segments, seg_size, block in SPECS:
        name = "${layer_fmt}".format(i=layer) + rel
        sd[name] = sd[name].index_select(dim, keep_index(segments, seg_size, block)).contiguous()

${py_asserts}
assert len(sd) == ${total}

out_dir.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out_dir / "model.safetensors"))
'''

T2_YAML = """# T2 for ${display}: remove attention head ${drop_head} from every layer.
# Each head-bearing tensor is rebuilt from the slices around the removed head.
# concat takes single tensor references only, so the per-layer concats are spelled out.
inputs:
  - model::${base_ref}
transforms:
${copies}
${deletes}
${concats}
${cleanup}
${asserts}
  - assert: { count: { of: '.*', is: ${total} } }
output: out/T2/model.safetensors
"""


def gen_t2(name: str, t: dict, drop_head: int) -> tuple[str, str, str]:
    fmt, glob, lre = t["layer_fmt"], t["layer_glob"], t["layer_re"]
    specs = t["head_tensors"]
    n_kept = t["n_heads"] - 1
    # task text
    head_list = []
    result_items = []
    check_items = []
    for k, s in enumerate(specs, start=1):
        head_list.append(f"- `{fmt.format(i='<i>')}{s['name']}`, shape `{s['shape']}`: {s['role']}.")
        slices = head_keep_slices(s, drop_head)
        axis = "rows" if s["dim"] == 0 else "columns"
        keep_txt = ", ".join(f"`{lo}..{hi - 1}`" for lo, hi in slices)
        shape = s["shape"].strip("[]").split(", ")
        shape[s["dim"]] = str(pruned_size(s))
        new_shape = "[" + ", ".join(shape) + "]"
        result_items.append(
            f"{k}. `{fmt.format(i='<i>')}{s['name']}` becomes `{new_shape}`: keep {axis} {keep_txt}, in that order."
        )
        check_items.append(f"- `{fmt.format(i=0)}{s['name']}` has shape `{new_shape}`;")
    task = render(
        T2_TASK, display=t["display"], n_heads_kept=n_kept, env=ENV_MD, base_desc=base_desc(t),
        total=t["total_tensors"], base_dtype=t["base_dtype"], n_layers=t["n_layers"], n_heads=t["n_heads"],
        head_dim=t["head_dim"], hidden=t["hidden"], layout_note=t["layout_note"], last_layer=t["n_layers"] - 1,
        head_tensor_list="\n".join(head_list), untouched=", ".join(t["head_untouched"]), drop_head=drop_head,
        result_items="\n".join(result_items), next_item=len(specs) + 1, out_item=len(specs) + 2,
        check_items="\n".join(check_items), target=name,
    )
    # python
    specs_py = "[\n" + "".join(
        f'    ("{s["name"]}", {s["dim"]}, {s["segments"]}, {s["seg_size"]}, {s["block"]}),\n' for s in specs
    ) + "]"
    py_asserts = []
    for s in specs:
        shape = [int(x) for x in s["shape"].strip("[]").split(", ")]
        shape[s["dim"]] = pruned_size(s)
        py_asserts.append(f'assert sd["{fmt.format(i=0)}{s["name"]}"].shape == {tuple(shape)}')
    py = render(
        T2_PY, display=t["display"], drop_head=drop_head, n_layers=t["n_layers"], specs=specs_py,
        base_ref=base_ref(t), layer_fmt=fmt, py_asserts="\n".join(py_asserts), total=t["total_tensors"],
    )
    # yaml
    copies, concats, asserts = [], [], []
    for s in specs:
        slices = head_keep_slices(s, drop_head)
        tmp = s["name"].replace(".", "_")
        for k, (lo, hi) in enumerate(slices):
            copies.append(
                f"  - copy: {{ from: '{lre}{esc(s['name'])}::{slice_expr(s['dim'], lo, hi)}', "
                f"to: '{fmt.format(i=chr(92) + '1')}tmp_{tmp}_{k}' }}"
            )
        for i in range(t["n_layers"]):
            pieces = ", ".join(f"{fmt.format(i=i)}tmp_{tmp}_{k}" for k in range(len(slices)))
            concats.append(
                f"  - concat: {{ from: [{pieces}], to: {fmt.format(i=i)}{s['name']}, dim: {s['dim']} }}"
            )
        shape = s["shape"].strip("[]").split(", ")
        shape[s["dim"]] = str(pruned_size(s))
        asserts.append(f"  - assert: {{ shape: {{ of: {fmt.format(i=0)}{s['name']}, is: [{', '.join(shape)}] }} }}")
    deletes = "  - delete: { target: '" + lre + "(" + "|".join(esc(s["name"]) for s in specs) + ")' }"
    cleanup = "  - delete: { target: '" + lre + r"tmp_.*' }"
    yaml = render(
        T2_YAML, display=t["display"], drop_head=drop_head, base_ref=base_ref(t), copies="\n".join(copies),
        deletes=deletes, concats="\n".join(concats), cleanup=cleanup, asserts="\n".join(asserts),
        total=t["total_tensors"],
    )
    return task, py, yaml


# --------------------------------------------------------------------------- T3

T3_TASK = """# T3: Mixed-precision export with sharding (${display})

## Objective

Prepare ${display} for a memory-constrained deployment: store the large
projection matrices in bfloat16, keep everything numerically sensitive
(embeddings, norms, biases) in float32, drop non-parameter buffers, and write
the result as a sharded checkpoint with an index file.

## Why it is meaningful

Exporting a checkpoint for deployment routinely mixes precisions: large
projection matrices in bfloat16 to halve the size, while embeddings, layer
norms and biases stay in float32 because they are small and numerically
sensitive. Sharding with an index file is what serving stacks expect. The
hazard is over-broad targeting: a pattern like `.*weight` also hits
embeddings and norms, and buffers such as causal masks are not parameters. A
correct solution has to cast exactly the intended matrices, drop the buffers,
upcast what must be float32, and produce a valid sharded layout.

${env}
## Input

- ${base_desc}: ${total} tensors, ${base_dtype}, ${n_layers} layers.
  Per layer `i` in 0..${last_layer} the projection matrices are
${proj_list}
${buffer_note}

## Required result

1. Cast exactly the ${n_matrices} projection matrices listed above to bfloat16
   (round-to-nearest-even, as `tensor.to(torch.bfloat16)` does${cast_note}).
2. Every other tensor is float32 in the output${upcast_note}. This includes
   ${keep_note}.
3. ${buffer_step}
4. Tensor names do not change.
5. Output: a sharded safetensors checkpoint in the directory `out/T3/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most ${shard_text} of tensor data, not
     counting file headers. A single tensor larger than that (here
     `${big_tensor}`, ${big_size}) is stored alone in its own shard.
   - Expected total: ${t3_total} tensors.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly ${n_matrices} tensors are bfloat16;
- `${probe_matrix}` is bfloat16;
- `${probe_keep}` is float32;
- the output has exactly ${t3_total} tensors.

## Grading

`grade.py T3 --target ${target}` compares `out/T3` with a hidden reference:
sharding rules, exact key set, shapes, dtypes and bit-exact values.
"""

T3_PY = '''"""T3 baseline for ${display}: bfloat16 projection matrices, float32 everything else, ${shard_text} shards."""

import re
import sys
from pathlib import Path

import torch

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T3")
MAX_SHARD = ${shard_bytes}

sd = load_checkpoint("${base_ref}")
matrix_re = re.compile(r"${layer_re}(${matrix_alt})")
buffer_re = re.compile(r"${layer_re}(${buffer_alt})")

out = {}
for name, tensor in sd.items():
    if buffer_re.fullmatch(name):
        continue
    out[name] = tensor.float().to(torch.bfloat16) if matrix_re.fullmatch(name) else tensor.float()

assert sum(t.dtype == torch.bfloat16 for t in out.values()) == ${n_matrices}
assert out["${probe_matrix}"].dtype == torch.bfloat16
assert out["${probe_keep}"].dtype == torch.float32
assert len(out) == ${t3_total}, len(out)

save_sharded_safetensors(out, out_dir, MAX_SHARD)
'''

T3_YAML = """# T3 for ${display}: bfloat16 projection matrices, float32 everything else, ${shard_text} shards.
inputs:
  - model::${base_ref}
transforms:
${buffer_lines}
  - cast_: { target: '.*', to: float32 }
  - assert: { count: { of: '${matrix_glob}', is: ${n_matrices} } }
  - cast_: { target: '${matrix_glob}', to: bfloat16 }
  - assert: { dtype: { of: ${probe_matrix}, is: bfloat16 } }
  - assert: { dtype: { of: ${probe_keep}, is: float32 } }
  - assert: { count: { of: '.*', is: ${t3_total} } }
output:
  path: out/T3
  format: safetensors
  shard: ${shard_bs}
"""


def gen_t3(name: str, t: dict, proj: list[tuple[str, str]]) -> tuple[str, str, str]:
    fmt, lre = t["layer_fmt"], t["layer_re"]
    n_matrices = t["n_layers"] * len(proj)
    t3_total = t["total_tensors"] - t["n_layers"] * len(t["buffers"])
    shard_bs, shard_bytes, shard_text = t["shard_t3"]
    proj_list = "\n".join(f"  - `{fmt.format(i='<i>')}{n}` (`{s}`)" for n, s in proj)
    if t["buffers"]:
        buffer_note = ("  Each layer also holds " + str(len(t["buffers"])) + " non-parameter buffer(s): " +
                       ", ".join(f"`{fmt.format(i='<i>')}{n}` (`{s}`, {d})" for n, s, d in t["buffers"]) + ".")
        buffer_step = ("Delete the " + str(t["n_layers"] * len(t["buffers"])) + " buffers " +
                       ", ".join(f"`{fmt.format(i='<i>')}{n}`" for n, _, _ in t["buffers"]) +
                       ". Do not delete any parameter.")
        buffer_alt = "|".join(esc(n) for n, _, _ in t["buffers"])
        buffer_lines = f"  - delete: {{ target: '{lre}({buffer_alt})' }}"
    else:
        buffer_note = "  There are no non-parameter buffers in this checkpoint."
        buffer_step = "There are no buffers to delete in this checkpoint; do not delete anything."
        buffer_alt = "(?!)"
        buffer_lines = "  # no buffers to delete in this checkpoint"
    fp16 = t["base_dtype"] != "float32"
    cast_note = "; the input is float16, so upcast to float32 first or cast directly, both give the same result" if fp16 else ""
    upcast_note = " (upcast from float16 where necessary; values are unchanged since float16 is exact in float32)" if fp16 else " with unchanged values"
    probe_matrix = fmt.format(i=0) + proj[0][0]
    probe_keep = t["nonlayer"][0][0]
    big_name, big_size = t["big_tensor"]
    task = render(
        T3_TASK, display=t["display"], env=ENV_MD, base_desc=base_desc(t), total=t["total_tensors"],
        base_dtype=t["base_dtype"], n_layers=t["n_layers"], last_layer=t["n_layers"] - 1, proj_list=proj_list,
        buffer_note=buffer_note, n_matrices=n_matrices, cast_note=cast_note, upcast_note=upcast_note,
        keep_note=t["keep_note"], buffer_step=buffer_step, shard_text=shard_text, big_tensor=big_name,
        big_size=big_size, t3_total=t3_total, probe_matrix=probe_matrix, probe_keep=probe_keep, target=name,
    )
    matrix_alt = "|".join(esc(n) for n, _ in proj)
    py = render(
        T3_PY, display=t["display"], shard_text=shard_text, shard_bytes=shard_bytes, base_ref=base_ref(t),
        layer_re=lre, matrix_alt=matrix_alt, buffer_alt=buffer_alt, n_matrices=n_matrices,
        probe_matrix=probe_matrix, probe_keep=probe_keep, t3_total=t3_total,
    )
    yaml = render(
        T3_YAML, display=t["display"], shard_text=shard_text, base_ref=base_ref(t), buffer_lines=buffer_lines,
        matrix_glob=f"{lre}({matrix_alt})", n_matrices=n_matrices, probe_matrix=probe_matrix,
        probe_keep=probe_keep, t3_total=t3_total, shard_bs=shard_bs,
    )
    return task, py, yaml


# --------------------------------------------------------------------------- T4

T4_TASK = """# T4: Task-vector merge of two fine-tunes (${display})

## Objective

Merge two fine-tunes of the same base model by task arithmetic: add a scaled
copy of each fine-tune's change (its task vector) to the base. Both fine-tunes
were trained with a frozen backbone, so only their MLP tensors differ from
the base; the merge must verify that assumption before touching anything.

## Why it is meaningful

Task arithmetic (adding scaled task vectors of several fine-tunes to a base)
is a widely used way to combine skills without retraining. The precondition
matters: it only makes sense if the fine-tunes share the base everywhere
except the tensors that were trained, so a careful merge verifies that before
touching anything. The arithmetic hazard is ordering: each task vector must be
taken against the unmodified base, not against a base that the first merge
already changed. A correct solution has to check three checkpoints against
each other, compute the merge in the right order, and leave everything else
untouched.

${env}
## Inputs

- ${base_desc}: the base, ${total} tensors, ${base_dtype}.
- `inputs/ft1/model.safetensors`: fine-tune 1, same ${total} names, shapes and dtypes.
- `inputs/ft2/model.safetensors`: fine-tune 2, same layout.

The MLP tensors are, per layer `i` in 0..${last_layer}:
${mlp_list}
${n_mlp} tensors in total.

## Required result

1. Before doing anything else, verify that the three checkpoints have the
   same tensor names and that every tensor outside the ${n_mlp} MLP tensors is
   identical in all three. Abort with an error if not.
2. For each of the ${n_mlp} MLP tensors `X`, with `lambda = 0.4`:

       out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

   computed in float32${dtype_note}.
3. Every other tensor is taken from the base unchanged. Tensor names do not
   change.
4. Output: a single file `out/T4/model.safetensors` with exactly ${total} tensors.

## Required checks

Your solution must fail loudly if any of these does not hold:

- the shared-tensor verification in step 1;
- exactly ${n_mlp} tensors were merged;
- the output has exactly ${total} tensors.

## Grading

`grade.py T4 --target ${target}` compares `out/T4` with a hidden reference:
exact key set, shapes, dtypes, bit-exact values for the ${n_other} unchanged
tensors, and for the ${n_mlp} merged tensors a relative Frobenius error of at
most ${tol} (so a different order of additions is fine).
"""

T4_PY = '''"""T4 baseline for ${display}: task-vector merge of two fine-tunes, lambda 0.4 each, MLP tensors only."""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T4")
LAMBDA = 0.4

base = load_checkpoint("${base_ref}")
ft1 = load_checkpoint("inputs/ft1/model.safetensors")
ft2 = load_checkpoint("inputs/ft2/model.safetensors")
mlp_re = re.compile(r"${layer_re}(${mlp_alt})")

assert set(base) == set(ft1) == set(ft2)
for name in base:
    if not mlp_re.fullmatch(name):
        assert torch.equal(base[name], ft1[name]), f"ft1 differs on shared tensor {name}"
        assert torch.equal(base[name], ft2[name]), f"ft2 differs on shared tensor {name}"

out = dict(base)
merged = 0
for name in base:
    if mlp_re.fullmatch(name):
        b, f1, f2 = base[name].float(), ft1[name].float(), ft2[name].float()
        ${merge_line}
        merged += 1
assert merged == ${n_mlp}
assert len(out) == ${total}

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
'''

T4_YAML = """# T4 for ${display}: task-vector merge of two fine-tunes, lambda 0.4 each, MLP tensors only.
inputs:
  - base::${base_ref}
  - ft1::inputs/ft1/model.safetensors
  - ft2::inputs/ft2/model.safetensors
transforms:
  - assert: { count: { of: 'ft1::.*', is: ${total} } }
  - assert: { count: { of: 'ft2::.*', is: ${total} } }
  - assert: { equal: { left: 'base::(?!${layer_re_nocap}(${mlp_alt})).+', right: 'ft1::\\g<0>' } }
  - assert: { equal: { left: 'base::(?!${layer_re_nocap}(${mlp_alt})).+', right: 'ft2::\\g<0>' } }
${upcast}
  # task vector 1: tv1 = 0.4 * (ft1 - base), taken against the unmodified base
  - copy: { from: 'ft1::${mlp_glob}', to: 'base::${mlp_to}.tv1' }
${cast_tv1}
  - subtract_: { from: 'base::${mlp_glob}', to: 'base::${mlp_to}.tv1' }
  - scale_: { target: 'base::${mlp_glob_nocap}\\.tv1', by: 0.4 }
  # task vector 2, also against the unmodified base (before tv1 is applied)
  - copy: { from: 'ft2::${mlp_glob}', to: 'base::${mlp_to}.tv2' }
${cast_tv2}
  - subtract_: { from: 'base::${mlp_glob}', to: 'base::${mlp_to}.tv2' }
  - scale_: { target: 'base::${mlp_glob_nocap}\\.tv2', by: 0.4 }
${apply}
  - delete: { target: 'base::${mlp_glob_nocap}\\.tv[12]' }
${downcast}
  - assert: { count: { of: 'base::.*', is: ${total} } }
output: out/T4/model.safetensors
"""


def gen_t4(name: str, t: dict, bug: bool = False) -> tuple[str, str, str]:
    fmt, lre = t["layer_fmt"], t["layer_re"]
    mlp = t["mlp_tensors"]
    n_mlp = t["n_layers"] * len(mlp)
    fp16 = t["base_dtype"] != "float32"
    tol = "1e-3" if fp16 else "1e-5"
    dtype_note = (", then cast back to float16 (the base dtype); the tolerance below absorbs the rounding" if fp16 else "")
    mlp_list = "\n".join(f"- `{fmt.format(i='<i>')}{n}` (`{s}`)" for n, s in mlp)
    task = render(
        T4_TASK, display=t["display"], env=ENV_MD, base_desc=base_desc(t), total=t["total_tensors"],
        base_dtype=t["base_dtype"], last_layer=t["n_layers"] - 1, mlp_list=mlp_list, n_mlp=n_mlp,
        dtype_note=dtype_note, n_other=t["total_tensors"] - n_mlp, tol=tol, target=name,
    )
    mlp_alt = "|".join(esc(n) for n, _ in mlp)
    cast_back = ".to(base[name].dtype)"
    if bug:
        merge_line = (f"tmp = b + LAMBDA * (f1 - b)\n        out[name] = (tmp + LAMBDA * (f2 - tmp)){cast_back}")
    else:
        merge_line = f"out[name] = (b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)){cast_back}"
    py = render(
        T4_PY, display=t["display"], base_ref=base_ref(t), layer_re=lre, mlp_alt=mlp_alt,
        merge_line=merge_line, n_mlp=n_mlp, total=t["total_tensors"],
    )
    # regex with one capture group for the layer and one for the tensor name
    lre_nocap = lre.replace("(\\d+)", "\\d+")
    mlp_glob = f"{lre}({mlp_alt})"
    mlp_glob_nocap = f"{lre_nocap}({mlp_alt})"
    mlp_to = fmt.format(i="\\1") + "\\2"
    apply_tv1 = f"  - add_: {{ from: 'base::{mlp_glob}\\.tv1', to: 'base::{mlp_to}' }}"
    apply_tv2 = f"  - add_: {{ from: 'base::{mlp_glob}\\.tv2', to: 'base::{mlp_to}' }}"
    if fp16:
        upcast = f"  - cast_: {{ target: 'base::{mlp_glob_nocap}', to: float32 }}"
        cast_tv1 = f"  - cast_: {{ target: 'base::{mlp_glob_nocap}\\.tv1', to: float32 }}"
        cast_tv2 = f"  - cast_: {{ target: 'base::{mlp_glob_nocap}\\.tv2', to: float32 }}"
        downcast = f"  - cast_: {{ target: 'base::{mlp_glob_nocap}', to: float16 }}"
    else:
        upcast = cast_tv1 = cast_tv2 = downcast = "  # base is float32 already"
    yaml = render(
        T4_YAML, display=t["display"], base_ref=base_ref(t), total=t["total_tensors"], layer_re_nocap=lre_nocap,
        mlp_alt=mlp_alt, upcast=upcast, mlp_glob=mlp_glob, mlp_to=mlp_to, cast_tv1=cast_tv1,
        mlp_glob_nocap=mlp_glob_nocap, cast_tv2=cast_tv2, apply=apply_tv1 + "\n" + apply_tv2, downcast=downcast,
    )
    if bug:
        # defect: tv1 is applied before tv2 is computed, so tv2 is taken against the modified base
        yaml = yaml.replace(
            "  # task vector 2, also against the unmodified base (before tv1 is applied)\n",
            apply_tv1 + "\n  # task vector 2\n",
        ).replace(apply_tv1 + "\n" + apply_tv2 + "\n", apply_tv2 + "\n")
        yaml = yaml.replace("tv1 = 0.4 * (ft1 - base), taken against the unmodified base", "tv1 = 0.4 * (ft1 - base)")
    return task, py, yaml


# --------------------------------------------------------------------------- T5

T5_TASK = """# T5: LoRA adapter merge with sharded export (${display})

## Objective

Fold a LoRA adapter into the base weights so that the result is a plain dense
checkpoint with no adapter tensors, then write it sharded. This is what
adapter frameworks call "merge and unload", done directly on the checkpoint
files.

## Why it is meaningful

Merging a LoRA adapter into the base weights ("merge and unload") is the last
step before deploying an adapted model without an adapter runtime. Doing it
directly on the checkpoint files avoids instantiating the model. Two details
decide correctness: the adapter's scaling factor alpha over r, and the
relation between the adapter's factor layout and the base weight layout,
which PEFT records as `fan_in_fan_out`. A correct solution has to map adapter
names to base names, multiply, scale, ${transpose_word}add, and leave no
adapter or intermediate tensor in a sharded output.

${env}
## Inputs

- ${base_desc}: the base, ${total} tensors, ${base_dtype}.
- `inputs/lora/adapter_model.safetensors`: a PEFT-style adapter with
  ${n_adapter} tensors, float32. For each layer `i` in 0..${last_layer} and each
  adapted module in ${module_list}:
  - `${peft_prefix}${layer_example}<module>.lora_A.weight`, shape `[${r}, ${in_dim}]`
  - `${peft_prefix}${layer_example}<module>.lora_B.weight`, shape `[${out_dim}, ${r}]`
- `inputs/lora/adapter_config.json`: `r = ${r}`, `lora_alpha = ${alpha}`,
  `target_modules = ${modules_json}`, `fan_in_fan_out = ${fifo_json}`.

The adapted base tensors are `${layer_example}<module>.weight`, shape
`${weight_shape}`. ${layout_sentence}

## Required result

For every layer `i` and every adapted module, with `A = lora_A.weight`,
`B = lora_B.weight` and `scale = lora_alpha / r = ${scale}`:

1. `${layer_example}<module>.weight += scale * ${product}`, computed in
   float32${dtype_note}. The result keeps its name, shape `${weight_shape}` and
   dtype ${base_dtype}.
2. No adapter tensor and no intermediate tensor appears in the output.
3. Every other base tensor is unchanged.
4. Output: a sharded safetensors checkpoint in the directory `out/T5/`:
   - shard files plus an index file `model.safetensors.index.json` whose
     `weight_map` maps every tensor name to the shard file that holds it;
   - the tensors in one shard total at most ${shard_text} of tensor data, not
     counting file headers. A single tensor larger than that (here
     `${big_tensor}`, ${big_size}) is stored alone in its own shard.
   - Expected total: ${total} tensors, the same names as the base.

## Required checks

Your solution must fail loudly if any of these does not hold before writing:

- exactly ${n_pairs} adapter pairs were found and merged;
- no tensor name containing `lora_` is in the output;
- `${probe}` still has shape `${weight_shape}`;
- the output has exactly ${total} tensors.

## Grading

`grade.py T5 --target ${target}` compares `out/T5` with a hidden reference:
sharding rules, exact key set, shapes, dtypes, bit-exact values for the
${n_other} unchanged tensors, and for the ${n_pairs} merged weights a relative
Frobenius error of at most ${tol}.
"""

T5_PY = '''"""T5 baseline for ${display}: merge a PEFT-style LoRA adapter into the base weights, ${shard_text} shards."""

import json
import sys
from pathlib import Path

from safetensors.torch import load_file

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T5")
N_LAYERS = ${n_layers}
MODULES = ${modules_py}
MAX_SHARD = ${shard_bytes}

sd = load_checkpoint("${base_ref}")
adapter = load_file("inputs/lora/adapter_model.safetensors")
config = json.loads(Path("inputs/lora/adapter_config.json").read_text())
scale = ${scale_expr}
transpose = config["fan_in_fan_out"]

merged = 0
for layer in range(N_LAYERS):
    for module in MODULES:
        prefix = "${peft_prefix}" + "${layer_fmt}".format(i=layer) + module
        a = adapter[f"{prefix}.lora_A.weight"]  # [r, in]
        b = adapter[f"{prefix}.lora_B.weight"]  # [out, r]
        key = "${layer_fmt}".format(i=layer) + module + ".weight"
        delta = b @ a
        if transpose:
            delta = delta.T
        sd[key] = (sd[key].float() + scale * delta).to(sd[key].dtype)
        merged += 1

assert merged == ${n_pairs}
assert not any("lora_" in name for name in sd)
assert sd["${probe}"].shape == ${weight_tuple}
assert len(sd) == ${total}

save_sharded_safetensors(sd, out_dir, MAX_SHARD)
'''

T5_YAML = """# T5 for ${display}: merge a PEFT-style LoRA adapter into the base weights, ${shard_text} shards.
# delta = (alpha / r) * ${product} with alpha=${alpha}, r=${r}.
inputs:
  - model::${base_ref}
  - lora::inputs/lora/adapter_model.safetensors
transforms:
  - assert: { count: { of: 'lora::.*lora_A\\.weight', is: ${n_pairs} } }
  - matmul:
      from_a: 'lora::${peft_esc}${layer_re}(${module_alt})\\.lora_B\\.weight'
      from_b: 'lora::${peft_prefix}${layer_to}.lora_A.weight'
      to: 'model::${layer_to}.delta${t_suffix}'
${permute}
${scale_line}
${upcast}
  - add_: { from: 'model::${layer_re}(${module_alt})\\.delta', to: 'model::${layer_to}.weight' }
${downcast}
  - delete: { target: 'model::${layer_re_nocap}(${module_alt})\\.delta(_t)?' }
  - assert: { not: { exists: 'model::.*lora_.*' } }
  - assert: { shape: { of: model::${probe}, is: ${weight_shape} } }
  - assert: { count: { of: 'model::.*', is: ${total} } }
output:
  path: out/T5
  format: safetensors
  shard: ${shard_bs}
"""


def gen_t5(name: str, t: dict, bug: bool = False) -> tuple[str, str, str]:
    fmt, lre = t["layer_fmt"], t["layer_re"]
    lo = t["lora"]
    n_pairs = t["n_layers"] * len(lo["modules"])
    fp16 = t["base_dtype"] != "float32"
    tol = "1e-3" if fp16 else "1e-5"
    scale = lo["alpha"] / lo["r"]
    shard_bs, shard_bytes, shard_text = t["shard_t5"]
    product = "(B @ A).T" if lo["fan_in_fan_out"] else "B @ A"
    layout_sentence = (
        "They use the Conv1D layout `[in, out]`, while the adapter factors follow the `nn.Linear` "
        "convention, which is what `fan_in_fan_out = true` signals: the low-rank product `B @ A` "
        "(shape `[out, in]`) must be transposed before it is added."
        if lo["fan_in_fan_out"] else
        "They use the `nn.Linear` layout `[out, in]`, the same convention as the adapter factors, so "
        "`B @ A` (shape `[out, in]`) is added without transposition (`fan_in_fan_out = false`)."
    )
    dtype_note = ", then cast back to float16 (the base dtype); the tolerance below absorbs the rounding" if fp16 else ""
    big_name, big_size = t["big_tensor"]
    probe = fmt.format(i=0) + lo["modules"][0] + ".weight"
    task = render(
        T5_TASK, display=t["display"], transpose_word="transpose, " if lo["fan_in_fan_out"] else "", env=ENV_MD,
        base_desc=base_desc(t), total=t["total_tensors"], base_dtype=t["base_dtype"], n_adapter=2 * n_pairs,
        last_layer=t["n_layers"] - 1, module_list=", ".join(f"`{m}`" for m in lo["modules"]),
        peft_prefix=lo["peft_prefix"], layer_example=fmt.format(i="<i>"), r=lo["r"], in_dim=lo["in"],
        out_dim=lo["out"], alpha=lo["alpha"], modules_json=json.dumps(lo["modules"]),
        fifo_json=json.dumps(lo["fan_in_fan_out"]), weight_shape=lo["weight_shape"], layout_sentence=layout_sentence,
        scale=f"{scale:g}", product=product, dtype_note=dtype_note, shard_text=shard_text, big_tensor=big_name,
        big_size=big_size, n_pairs=n_pairs, probe=probe, target=name, n_other=t["total_tensors"] - n_pairs, tol=tol,
    )
    weight_tuple = tuple(int(x) for x in lo["weight_shape"].strip("[]").split(", "))
    py = render(
        T5_PY, display=t["display"], shard_text=shard_text, n_layers=t["n_layers"], modules_py=json.dumps(lo["modules"]),
        shard_bytes=shard_bytes, base_ref=base_ref(t),
        scale_expr="1.0  # defect: alpha / r forgotten" if bug else 'config["lora_alpha"] / config["r"]',
        peft_prefix=lo["peft_prefix"], layer_fmt=fmt, n_pairs=n_pairs, probe=probe, weight_tuple=weight_tuple,
        total=t["total_tensors"],
    )
    module_alt = "|".join(esc(m) for m in lo["modules"])
    lre_nocap = lre.replace("(\\d+)", "\\d+")
    layer_to = fmt.format(i="\\1") + "\\2"
    t_suffix = "_t" if lo["fan_in_fan_out"] else ""
    permute = (
        f"  - permute: {{ from: 'model::{lre}({module_alt})\\.delta_t', to: 'model::{layer_to}.delta', order: [1, 0] }}"
        if lo["fan_in_fan_out"] else "  # nn.Linear layout: no transpose needed"
    )
    scale_line = ("  # defect: the alpha / r scale is missing" if bug
                  else f"  - scale_: {{ target: 'model::{lre_nocap}({module_alt})\\.delta', by: {scale:g} }}")
    if fp16:
        upcast = f"  - cast_: {{ target: 'model::{lre_nocap}({module_alt})\\.weight', to: float32 }}"
        downcast = f"  - cast_: {{ target: 'model::{lre_nocap}({module_alt})\\.weight', to: float16 }}"
    else:
        upcast = downcast = "  # base is float32 already"
    yaml = render(
        T5_YAML, display=t["display"], shard_text=shard_text, product=product, alpha=lo["alpha"], r=lo["r"],
        base_ref=base_ref(t), n_pairs=n_pairs, peft_esc=esc(lo["peft_prefix"]), layer_re=lre, module_alt=module_alt,
        peft_prefix=lo["peft_prefix"], layer_to=layer_to, t_suffix=t_suffix, permute=permute, scale_line=scale_line,
        upcast=upcast, downcast=downcast, layer_re_nocap=lre_nocap, probe=probe, weight_shape=lo["weight_shape"],
        total=t["total_tensors"], shard_bs=shard_bs,
    )
    return task, py, yaml


# --------------------------------------------------------------------------- driver

DEFECTS = {
    "T1": "The drop set is off by one: the last dropped block is the one after the specified block "
          "(for example blocks 2, 5, 9 instead of 2, 5, 8). Block counts still match, so the artifact's "
          "own checks pass.",
    "T2": "The removed head is off by one (head N+1 instead of head N). All shapes are still correct, so "
          "the artifact's own shape checks pass.",
    "T3": "One projection family is missing from the bfloat16 cast pattern (the attention output "
          "projection stays float32); the artifact's own count check was adjusted to the wrong number.",
    "T4": "Task vector 2 is computed against the base after task vector 1 has already been added, so "
          "the second delta is taken from the wrong reference point.",
    "T5": "The adapter scaling factor alpha / r is missing: the raw low-rank product is added.",
}


def main() -> int:
    answers_by_target: dict[str, dict] = {}
    for tname, t in TARGETS.items():
        sol_p = HERE / "solutions" / tname / "P"
        sol_b = HERE / "solutions" / tname / "B"
        rev_p = HERE / "review" / tname / "P"
        rev_b = HERE / "review" / tname / "B"
        for d in (sol_p, sol_b, rev_p, rev_b):
            d.mkdir(parents=True, exist_ok=True)
        (sol_p / "_ckpt.py").write_text(IO_PY)
        (rev_p / "_ckpt.py").write_text(IO_PY)

        drop = t["drop_layers"]
        drop_bug = drop[:-1] + [drop[-1] + 1]
        head_bug = t["drop_head"] + 1
        proj_bug = [p for p in t["proj_matrices"] if p[0] not in ("attn.c_proj.weight", "self_attn.o_proj.weight", "attention.dense.weight")]

        good = {
            "T1": gen_t1(tname, t, drop),
            "T2": gen_t2(tname, t, t["drop_head"]),
            "T3": gen_t3(tname, t, t["proj_matrices"]),
            "T4": gen_t4(tname, t),
            "T5": gen_t5(tname, t),
        }
        bad = {
            "T1": gen_t1(tname, t, drop_bug),
            "T2": gen_t2(tname, t, head_bug),
            "T3": gen_t3(tname, t, proj_bug),
            "T4": gen_t4(tname, t, bug=True),
            "T5": gen_t5(tname, t, bug=True),
        }
        for test, (task, py, yaml) in good.items():
            (HERE / "tasks" / f"{test}-{TESTS[test]}" / f"TASK-{tname}.md").write_text(task)
            (sol_p / f"{test}.py").write_text(py)
            (sol_b / f"{test}.yaml").write_text(yaml)
        for test, (_task, py, yaml) in bad.items():
            (rev_p / f"{test}-defective.py").write_text(py)
            (rev_b / f"{test}-defective.yaml").write_text(yaml)
        answers_by_target[tname] = {test: DEFECTS[test] for test in TESTS}
        (HERE / "review" / tname / "answers.json").write_text(json.dumps(answers_by_target[tname], indent=2) + "\n")
        print(f"[generate] {tname}: 5 tasks, 5+5 solutions, 5+5 review artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
