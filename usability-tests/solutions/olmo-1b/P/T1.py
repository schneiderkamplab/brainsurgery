"""T1 baseline for OLMo-1B-0724-hf: remove blocks 2, 6, 10, 14 and renumber the rest contiguously."""

import re
import sys
from pathlib import Path

from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T1")
DROP = {2, 6, 10, 14}
N_LAYERS = 16

sd = load_checkpoint("inputs/base")
layer_re = re.compile(r"model\.layers\.(\d+)\.(.+)")
kept = [i for i in range(N_LAYERS) if i not in DROP]
renumber = {old: new for new, old in enumerate(kept)}

out = {}
for name, tensor in sd.items():
    match = layer_re.fullmatch(name)
    if match is None:
        out[name] = tensor
        continue
    old = int(match.group(1))
    if old in DROP:
        continue
    out["model.layers.{i}.".format(i=renumber[old]) + match.group(2)] = tensor

layers = {int(m.group(1)) for n in out if (m := layer_re.fullmatch(n))}
assert layers == set(range(len(kept))), sorted(layers)
assert len(out) == 86, len(out)

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
