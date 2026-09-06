# Suggested paper language

## Result

We evaluated correctness with two independently checked protocols. First, ten
hand-verifiable cases covered identity serialization, structural edits,
metamorphic rename and split/concatenate inversions, arithmetic, dtype
conversion, and sharded save/reload. All 161 tensor--case comparisons matched
the PyTorch oracle exactly, and all 149 tensors outside the declared write-sets
were byte-identical to their inputs. Second, we applied an explicit
identity-preserving transformation to pinned GPT-2 124M, OLMo 1B, and Pythia 1B
checkpoints. All 518 output tensors matched the independently loaded inputs
exactly, including all 515 tensors outside the three one-tensor write-sets. All
source checkpoint files remained unchanged. Deliberately corrupted value,
dtype, and key-set controls were detected by the verifier.

## Claim

These results show exact tensor-state correctness and no unintended tensor
changes for the enumerated transformations, checkpoint layouts, and revisions.
They do not constitute a proof for arbitrary plans or formats.

## Limitation

BrainSurgery currently operates on tensor state dictionaries. Its safetensors
outputs did not retain custom input header metadata, and arbitrary checkpoint
sidecar files are not copied by this interface. We therefore restrict the claim
to tensor-state preservation rather than complete container-level information
preservation.
