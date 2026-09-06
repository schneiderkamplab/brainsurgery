# Checkpoint-scaling result

Protocol: `eacl2027_scaling_v1`

Run: `mac_scaling_preflight_2026_09_06_v11`

Commit: `299671ec49bb4c5c2705c92680ed82f281ef3bd8`

Status: **NON-REPORTABLE PREFLIGHT**

| Model | Method | Correct | Wall median (s) | Peak RSS (GiB) | Effective GiB/s | Output shards |
|---|---|---:|---:|---:|---:|---:|
| deterministic tiny fixture | python_pytorch | 1/1 | — | — | — | 1 |
| deterministic tiny fixture | brainsurgery_inmemory | 1/1 | — | — | — | 1 |
| deterministic tiny fixture | brainsurgery_arena | 1/1 | — | — | — | 1 |

Performance values are suppressed: run was explicitly requested as a synthetic smoke preflight; reported systems run requires Linux; Git worktree is dirty; reported evaluation requires the complete ordered ten-checkpoint matrix; correct, fully sampled measured repetitions are incomplete.

Frozen CPU checkpoint rewrite on the enumerated revisions and one recorded Linux system; not GPU, training, inference, usability, or general tool superiority evidence.
