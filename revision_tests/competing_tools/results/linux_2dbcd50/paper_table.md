# Competing-tool run (reported-size candidate)

**REPORTABLE CANDIDATE: all automated eligibility gates passed.**

| Case | Tool | Correct runs | Median wall (s) | Median peak RSS (MiB) | Output (MiB) | Spec lines |
|---|---|---:|---:|---:|---:|---:|
| R01 | brainsurgery | 5/5 | 5.042380 | 1785.69 | 522.72 | 7 |
| R01 | torch_state_bridge | 5/5 | 1.478026 | 1571.89 | 522.71 | 2 |
| M01 | brainsurgery | 5/5 | 5.292274 | 2166.67 | 474.72 | 17 |
| M01 | mergekit | 5/5 | 4.887094 | 2890.42 | 474.72 | 11 |
| M02 | brainsurgery | 5/5 | 5.479855 | 3115.05 | 474.72 | 32 |
| M02 | mergekit | 5/5 | 5.289435 | 3895.29 | 474.72 | 13 |

| Case | Competitor | BrainSurgery / competitor median wall ratio |
|---|---|---:|
| M01 | mergekit | 1.0829 |
| M02 | mergekit | 1.0360 |
| R01 | torch_state_bridge | 3.4116 |

Timings include process startup, input loading, transformation, and output save. Specification lines are descriptive only and are not a usability measure.
