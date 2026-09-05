"""T1 baseline for GPT-2 (124M): remove blocks 2, 5, 9 and renumber the rest contiguously."""

import re
import sys
from pathlib import Path

from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T1")
DROP = {2, 5, 9}
N_LAYERS = 12

sd = load_checkpoint("inputs/base/model.safetensors")
layer_re = re.compile(r"h\.(\d+)\.(.+)")
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
    out["h.{i}.".format(i=renumber[old]) + match.group(2)] = tensor

layers = {int(m.group(1)) for n in out if (m := layer_re.fullmatch(n))}
assert layers == set(range(len(kept))), sorted(layers)
assert len(out) == 121, len(out)

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
