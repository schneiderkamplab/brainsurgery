# Robustness and failure-semantics result

Protocol: `eacl2027_robustness_v1`
Run: `eacl2027_robustness_linux_2dbcd50`
Commit: `2dbcd505115100f892e906413076ae93b3fcaa16`

| Case | Failure class | Process | Diagnostic | Input unchanged | Artifact | Safe | Evaluation |
|---|---|---|---:|---:|---|---:|---|
| R01 | plan validation | failure | yes | yes | absent | yes | PASS |
| R02 | plan validation | failure | yes | yes | absent | yes | PASS |
| R03 | plan validation | failure | yes | yes | absent | yes | PASS |
| R04 | plan validation | failure | yes | yes | absent | yes | PASS |
| R05 | tensor selection | failure | yes | yes | absent | yes | PASS |
| R06 | tensor selection | failure | yes | yes | absent | yes | PASS |
| R07 | tensor selection | failure | yes | yes | absent | yes | PASS |
| R08 | reference validation | failure | yes | yes | absent | yes | PASS |
| R09 | input validation | failure | yes | yes | absent | yes | PASS |
| R10 | runtime guard | failure | yes | yes | absent | yes | PASS |
| R11 | input integrity | failure | yes | yes | absent | yes | PASS |
| R12 | input integrity | failure | yes | yes | absent | yes | PASS |
| R13 | input integrity | failure | yes | yes | absent | yes | PASS |
| R14 | destination preservation | failure | yes | yes | preexisting unchanged | yes | PASS |
| R15 | publication failure | failure | yes | yes | absent | yes | PASS |
| R16 | publication failure | failure | yes | yes | partial without index | no | PASS |
| R17 | interruption | interrupted | n/a | yes | partial without index | no | PASS |
| R18 | destination preservation | failure | yes | yes | preexisting changed with unindexed shard | no | PASS |
| R19 | positive control | success | n/a | yes | valid complete | yes | PASS |

## Aggregate

- Evaluation cases: 19/19 passed.
- Expected non-success outcomes: 18/18 detected.
- Applicable diagnostics: 17/17 matched.
- Source inputs: 19/19 unchanged.
- Failures before publication: 15/15 withheld output or preserved the destination.
- Pre-existing destinations: 1/2 preserved.
- Observed-safe cases: 16/19.
- Partial or mixed-output findings: 3 (R16, R17, R18).

## Claim boundary

The result characterizes 19 deterministic cases using the in-memory provider, safetensors, and the recorded local filesystem. It supports rejection, source-preservation, and diagnostic claims only for these cases. Because injected save failure and interruption expose partial or mixed shard directories, it does not support transactional or crash-safe publication claims.
