# Correctness and preservation results

Two complementary protocols were executed through the public BrainSurgery CLI
on macOS. Neither protocol imports Axon or Synapse.

## Results

| Evaluation | Cases passed | Oracle tensor-case comparisons exact | Untouched tensor-case comparisons exact | Source checks |
|---|---:|---:|---:|---:|
| Hand-verifiable primitive cases | 10/10 | 161/161 | 149/149 | 10/10 |
| Pinned real checkpoints | 3/3 | 518/518 | 515/515 | 3/3 |
| **Combined** | **13/13** | **679/679** | **664/664** | **13/13** |

The primitive cases cover identity serialization, rename/inverse rename, copy,
move, delete, split/concatenate inversion, add/subtract/multiply/scale,
same-dtype cast, float32-to-bfloat16 conversion against an independent PyTorch
oracle, and a three-shard save/reload cycle. All five verifier controls passed,
including deliberately corrupted value, dtype, and tensor-name-set controls.

The real-checkpoint protocol covers these exact pinned revisions:

| Checkpoint | Revision | Tensors exact | Untouched exact |
|---|---|---:|---:|
| GPT-2 124M | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 160/160 | 159/159 |
| OLMo 1B, two input shards | `d7cbab742d80589e714b1a2d7f838dcd21cbe143` | 114/114 | 113/113 |
| Pythia 1B | `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` | 244/244 | 243/243 |

The Hugging Face cache metadata matched every pinned revision, and every source
checkpoint data/index file remained byte-identical.

## Claim supported

> For the enumerated lossless transformations and checkpoint revisions,
> BrainSurgery produced the independently expected tensor state exactly and
> preserved every tensor outside the declared write-set byte for byte.

This is reproducible evidence, not a proof for arbitrary programs or checkpoint
formats.

## Important limitation

Custom safetensors header metadata was preserved in 0/13 cases. BrainSurgery's
current interface transforms and serializes tensor state dictionaries; it does
not copy arbitrary safetensors header metadata or sidecar files. Consequently,
the paper must say **exact tensor-state preservation**, not “no information is
ever lost” or “the complete checkpoint container is preserved.”

Runtime from these Mac executions is not used as systems-performance evidence.

## Evidence

- [`2026_09_06_macos_30adb808/`](2026_09_06_macos_30adb808/paper_table.md):
  controlled primitive result, protocol `eacl2027_correctness_v1`.
- [`2026_09_06_real_macos_c5e464b9/`](2026_09_06_real_macos_c5e464b9/paper_table.md):
  real-checkpoint result, protocol `eacl2027_real_preservation_v1`.
- [`paper_table.tex`](paper_table.tex): compact LaTeX table for the paper.
- [`paper_text.md`](paper_text.md): conservative result language and limitation.

Raw output checkpoints, actual plans, captured logs, and per-case records remain
under the ignored `log/revision_tests/` run directories named in each compact
summary.
