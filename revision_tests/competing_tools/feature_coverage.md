# Related-system feature coverage

Status: protocol-ready; Linux results and the official usability cohort are
still pending.

This table separates direct, operation-matched comparisons from adjacent
capabilities. It is a claim map, not a count of every feature exposed by any
package. A capability is called **direct** only when BrainSurgery and the named
tool receive the same tensor contract and their outputs are checked by the same
independent oracle.

| Capability | Most relevant comparison | Evidence status | Evidence or disposition |
|---|---|---|---|
| Regex/capture key rewriting | `torch-state-bridge` | Direct comparison prepared | R01; exact names, dtypes, shapes, and tensor bytes |
| Two-checkpoint weighted merge | MergeKit | Direct comparison prepared | M01; shared float32 contract and independent numerical oracle |
| Base-relative task-vector arithmetic | MergeKit | Direct comparison prepared | M02; shared float32 contract and independent numerical oracle |
| Block deletion and contiguous reindexing | MergeKit layer selection/slicing | Usability protocol prepared; not a direct tool benchmark | Usability T1 compares BrainSurgery, Python/PyTorch, and an allowed-package condition |
| Tensor slicing and concatenation | MergeKit slicing is adjacent, not identical | Correctness tested; usability protocol prepared | Correctness C06 and usability T2; do not label this a MergeKit head-pruning comparison |
| Mixed-precision conversion and sharded safetensors export | PyTorch/safetensors | Operation-matched systems protocol prepared | Usability T3 plus the scaling protocol; Linux measurements pending |
| LoRA merge and dense sharded export | PEFT and MergeKit's adjacent LoRA functionality | Usability protocol prepared; no fixed-tool benchmark | Usability T5; report the package actually selected in condition F rather than implying a MergeKit comparison |
| File-backed/out-of-core execution | Direct PyTorch in-memory baseline; MergeKit has adjacent out-of-core functionality | Operation-matched systems protocol prepared | Scaling compares Python/PyTorch, BrainSurgery in-memory, and BrainSurgery arena; it does not benchmark MergeKit's out-of-core implementation |
| MoE construction/upcycling | MergeKit | Not evaluated | Outside the present revision unless a downstream-quality protocol is added |
| Distributed checkpoint resharding, rank-local state, and optimizer state | Orbax and PyTorch Distributed Checkpoint | Not evaluated | Deferred; exclude from evaluated-capability claims |

## Count that may be reported

- **3 distinct operations** have frozen direct comparisons against named
  competing tools: two against MergeKit and one against
  `torch-state-bridge`.
- Those three operations produce **6 tool/case pairings** because each is run
  through BrainSurgery and through its comparator in every repetition.
- The remaining rows are complementary correctness, usability, or systems
  evidence. They must not be added to the direct-comparison count.
- Orbax is related-work positioning, not an executable baseline in this
  revision. No zero, failure, or unsupported score should be assigned to it.

The corresponding compact LaTeX table is in `feature_coverage.tex`. Update
both files together if an operation is added or its evidence status changes.
