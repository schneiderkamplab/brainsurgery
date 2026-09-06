# Competing-tool comparisons

Compare only operations for which the tools have genuine overlapping scope.
Initial candidates are MergeKit and `torch-state-bridge`; Orbax should be
covered in positioning unless an equivalent executable operation can be
defined.

For each comparison, define one tool-neutral transformation specification and
validate every output with the same independent oracle. Record installation
versions, input format, unsupported features, wall time, peak memory, output
equivalence, required code/configuration size, and failures. Dependency or
format incompatibility is a result and must not be hidden by hand-editing an
output.

The usability study's condition F is separate: it measures an agent choosing
from allowed packages, whereas this area is a controlled systems comparison.
