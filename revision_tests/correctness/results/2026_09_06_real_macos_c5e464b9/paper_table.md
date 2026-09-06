# Real-checkpoint preservation result

Protocol: `eacl2027_real_preservation_v1`
Run: `2026_09_06_real_preservation_macos_c5e464b9`
Commit: `c5e464b9a06c50c7549ccae26d6f098f7b25822b`

| Target | Revision | Tensors exact | Untouched exact | Sources unchanged | Result |
|---|---|---:|---:|---:|---|
| gpt_2_124m | `607a30d783df` | 160/160 | 159/159 | yes | PASS |
| olmo_1b_0724_hf | `d7cbab742d80` | 114/114 | 113/113 | yes | PASS |
| pythia_1b | `f73d7dcc545c` | 244/244 | 243/243 | yes | PASS |

## Aggregate

- Checkpoints: 3/3 passed.
- Tensors: 518/518 exact.
- Untouched tensors: 515/515 exact.
- Source checkpoint sets: 3/3 unchanged.
- Custom safetensors metadata preserved: 0/3 cases (secondary observation).
- Runtime is intentionally not reported as a performance result.

## Claim boundary

Exact tensor-state preservation for an explicit identity operation on the three enumerated checkpoint revisions; not a performance measurement.
