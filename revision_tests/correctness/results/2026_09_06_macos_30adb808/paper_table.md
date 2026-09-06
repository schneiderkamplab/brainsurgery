# Correctness and preservation result

Protocol: `eacl2027_correctness_v1`
Run: `2026_09_06_correctness_macos_30adb808`
Commit: `30adb808fff539374dd567ebee67d385decd8b7a`

| Case | Operation | Class | Oracle exact | Untouched exact | Input unchanged | Result |
|---|---|---|---:|---:|---:|---|
| C01 | identity serialization | lossless | 16/16 | 15/15 | yes | PASS |
| C02 | rename inverse | lossless metamorphic | 16/16 | 15/15 | yes | PASS |
| C03 | copy | lossless structural | 17/17 | 16/16 | yes | PASS |
| C04 | move | lossless structural | 16/16 | 15/15 | yes | PASS |
| C05 | delete | lossless structural | 15/15 | 15/15 | yes | PASS |
| C06 | split concat inverse | lossless metamorphic | 16/16 | 15/15 | yes | PASS |
| C07 | arithmetic | exact arithmetic oracle | 17/17 | 13/13 | yes | PASS |
| C08 | same dtype cast | lossless | 16/16 | 15/15 | yes | PASS |
| C09 | float32 to bfloat16 | intentionally lossy oracle | 16/16 | 15/15 | yes | PASS |
| C10 | sharded save reload | lossless serialization | 16/16 | 15/15 | yes | PASS |

## Aggregate

- Cases: 10/10 passed.
- Oracle tensors: 161/161 exact.
- Untouched tensor checks: 149/149 exact.
- Source checkpoint checks: 10/10 unchanged.
- Verifier controls: 5/5 passed.
- Safetensors custom metadata preserved: 0/10 cases (secondary observation, not a primary endpoint).

## Claim boundary

Exact tensor-state correctness for the enumerated cases; custom safetensors metadata and arbitrary sidecar files are not covered by the primary claim.
