# Checkpoint-scaling result

Protocol: `eacl2027_scaling_v1`  
Run: `eacl2027_scaling_linux_2dbcd50`  
Commit: `2dbcd505115100f892e906413076ae93b3fcaa16`  
Status: **REPORTABLE CANDIDATE**

| Model | Method | Correct | Wall median (s) | Peak RSS (GiB) | Effective GiB/s | Output shards |
|---|---|---:|---:|---:|---:|---:|
| Pythia 70M | python_pytorch | 5/5 | 1.169 | 0.768 | 0.265 | 1 |
| Pythia 70M | brainsurgery_inmemory | 5/5 | 4.335 | 0.815 | 0.071 | 1 |
| Pythia 70M | brainsurgery_arena | 5/5 | 4.383 | 0.878 | 0.071 | 1 |
| Pythia 410M | python_pytorch | 5/5 | 1.623 | 2.101 | 1.046 | 2 |
| Pythia 410M | brainsurgery_inmemory | 5/5 | 5.227 | 1.520 | 0.325 | 2 |
| Pythia 410M | brainsurgery_arena | 5/5 | 5.086 | 2.320 | 0.334 | 2 |
| Pythia 2.8B | python_pytorch | 5/5 | 4.297 | 10.960 | 2.464 | 12 |
| Pythia 2.8B | brainsurgery_inmemory | 5/5 | 9.057 | 5.965 | 1.169 | 11 |
| Pythia 2.8B | brainsurgery_arena | 5/5 | 9.257 | 11.206 | 1.144 | 11 |
| Pythia 12B | python_pytorch | 5/5 | 15.502 | 44.765 | 2.865 | 56 |
| Pythia 12B | brainsurgery_inmemory | 5/5 | 24.589 | 22.878 | 1.806 | 57 |
| Pythia 12B | brainsurgery_arena | 5/5 | 26.537 | 28.184 | 1.674 | 57 |
| GPT-2 124M | python_pytorch | 5/5 | 1.421 | 1.471 | 0.719 | 2 |
| GPT-2 124M | brainsurgery_inmemory | 5/5 | 5.070 | 1.181 | 0.201 | 2 |
| GPT-2 124M | brainsurgery_arena | 5/5 | 5.047 | 1.609 | 0.202 | 2 |
| GPT-2 XL 1.5B | python_pytorch | 5/5 | 5.024 | 12.288 | 2.384 | 13 |
| GPT-2 XL 1.5B | brainsurgery_inmemory | 5/5 | 9.941 | 6.662 | 1.205 | 13 |
| GPT-2 XL 1.5B | brainsurgery_arena | 5/5 | 10.772 | 12.499 | 1.112 | 13 |
| OLMo 1B (sharded) | python_pytorch | 5/5 | 4.514 | 9.649 | 2.112 | 10 |
| OLMo 1B (sharded) | brainsurgery_inmemory | 5/5 | 9.700 | 5.439 | 0.983 | 10 |
| OLMo 1B (sharded) | brainsurgery_arena | 5/5 | 9.348 | 9.367 | 1.020 | 10 |
| OLMo 7B (sharded, FP32) | python_pytorch | 5/5 | 18.395 | 30.806 | 2.790 | 66 |
| OLMo 7B (sharded, FP32) | brainsurgery_inmemory | 5/5 | 24.981 | 26.330 | 2.054 | 65 |
| OLMo 7B (sharded, FP32) | brainsurgery_arena | 5/5 | 30.235 | 29.023 | 1.697 | 65 |
| Qwen2.5 0.5B | python_pytorch | 5/5 | 1.792 | 2.338 | 1.027 | 2 |
| Qwen2.5 0.5B | brainsurgery_inmemory | 5/5 | 5.390 | 1.591 | 0.341 | 2 |
| Qwen2.5 0.5B | brainsurgery_arena | 5/5 | 5.358 | 2.346 | 0.344 | 2 |
| Qwen2.5 7B (sharded, BF16) | python_pytorch | 5/5 | 10.795 | 28.868 | 2.628 | 30 |
| Qwen2.5 7B (sharded, BF16) | brainsurgery_inmemory | 5/5 | 16.094 | 14.856 | 1.763 | 31 |
| Qwen2.5 7B (sharded, BF16) | brainsurgery_arena | 5/5 | 17.963 | 18.036 | 1.579 | 31 |

Frozen CPU checkpoint rewrite on the enumerated revisions and one recorded Linux system; not GPU, training, inference, usability, or general tool superiority evidence.
