# Ten-checkpoint CUDA behavioral regression

Protocol: `eacl2027_behavioral_matrix_v2`  
Run: `eacl2027_behavioral_matrix_cuda_99693f2`  
Commit: `99693f26716da7ffbe4109b20bcdcd475c008022`  
GPU: NVIDIA B200  
Status: **REPORTABLE CANDIDATE — HUMAN ANONYMITY REVIEW REQUIRED**

| Model | Parameters | Dtype | Exact tensors | Exact logits | Top-1 | Greedy | MCQ agreement |
|---|---:|---|---:|---:|---:|---:|---:|
| Pythia 70M | 70M | float16 | 94/94 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 410M | 410M | float16 | 364/364 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 2.8B | 2.80B | float16 | 484/484 | 70/70 | 70/70 | 70/70 | 60/60 |
| Pythia 12B | 12.00B | float16 | 544/544 | 70/70 | 70/70 | 70/70 | 60/60 |
| GPT-2 124M | 124M | float32 | 160/160 | 70/70 | 70/70 | 70/70 | 60/60 |
| GPT-2 XL 1.5B | 1.50B | float32 | 628/628 | 70/70 | 70/70 | 70/70 | 60/60 |
| OLMo 1B (sharded) | 1.00B | float32 | 114/114 | 70/70 | 70/70 | 70/70 | 60/60 |
| OLMo 7B (sharded, FP32) | 7.00B | float32 | 226/226 | 70/70 | 70/70 | 70/70 | 60/60 |
| Qwen2.5 0.5B | 500M | bfloat16 | 290/290 | 70/70 | 70/70 | 70/70 | 60/60 |
| Qwen2.5 7B (sharded, BF16) | 7.00B | bfloat16 | 339/339 | 70/70 | 70/70 | 70/70 | 60/60 |

Aggregate: 3243/3243 tensors were byte-exact and all 700/700 prompt pairs had exact final-token logits.

Lossless regression evidence for the enumerated checkpoints, prompt manifest, software environment, and GPU; not evidence of general model quality.
