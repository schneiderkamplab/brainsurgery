# Competing-tool preflight interpretation

The actual-package macOS preflight produced correct outputs in all 6/6 measured
tool/case attempts after all six warm-ups passed. The rename case preserved the
160 GPT-2 source tensors and one synthetic nonmatching marker byte-exactly for
both BrainSurgery and `torch-state-bridge`. BrainSurgery and MergeKit each
matched the independent float32 oracle on all 148 mutually supported GPT-2
parameters in both arithmetic cases; the maximum absolute difference over
those four outputs was `2.9802322387695312e-08`, below the frozen
`atol=rtol=1e-6` rule.

This preflight establishes only that the frozen adapters, shared-operation
contracts, and independent oracle execute together. It does not support a
runtime, memory, efficiency, scaling, usability, downstream-quality, or general
tool-ranking claim. Those statements must use the controlled Linux run, and
only if that run passes every reporting gate.
