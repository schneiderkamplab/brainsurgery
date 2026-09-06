from pathlib import Path
import re

import torch
from safetensors.torch import load_file, save_file


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    source = Path('inputs/base/model.safetensors')
    destination = Path('out/T1/model.safetensors')
    temporary = destination.with_suffix('.tmp')
    require(not destination.exists(), 'Output already exists')
    state = load_file(str(source), device='cpu')
    block_pattern = re.compile(r'h\.(\d+)\.(.+)')
    survivors = [0, 1, 3, 4, 6, 7, 9, 10, 11]
    mapping = {old: new for new, old in enumerate(survivors)}
    nonblocks = {'wte.weight', 'wpe.weight', 'ln_f.weight', 'ln_f.bias'}
    require(len(state) == 160, 'Expected 160 input tensors')
    counts = {i: 0 for i in range(12)}
    output = {}
    origins = {}
    for key, tensor in state.items():
        match = block_pattern.fullmatch(key)
        if match:
            old = int(match[1])
            require(old in counts, f'Unexpected block: {key}')
            counts[old] += 1
            if old not in mapping:
                continue
            new_key = f'h.{mapping[old]}.{match[2]}'
        else:
            require(key in nonblocks, f'Unexpected non-block tensor: {key}')
            new_key = key
        require(new_key not in output, f'Collision: {new_key}')
        output[new_key] = tensor
        origins[new_key] = key
    require(all(n == 13 for n in counts.values()), 'Expected 13 tensors per input block')
    require(nonblocks <= output.keys(), 'Missing non-block tensors')
    indices = {int(m[1]) for k in output if (m := block_pattern.fullmatch(k))}
    require(not indices.intersection({9, 10, 11}), 'Blocks 9, 10, or 11 remain')
    require(indices == set(range(9)), 'Expected exactly blocks 0 through 8')
    require(sum(bool(re.fullmatch(r'h\.\d+\.attn\.c_attn\.weight', k))
                for k in output) == 9, 'Expected nine attention projection weights')
    require(len(output) == 121, 'Expected exactly 121 output tensors')

    # Validate serialization before publishing the final checkpoint.
    try:
        save_file(output, str(temporary))
        restored = load_file(str(temporary), device='cpu')
        require(restored.keys() == output.keys(), 'Serialized key mismatch')
        for key, tensor in restored.items():
            original = state[origins[key]]
            require(tensor.shape == original.shape and tensor.dtype == original.dtype,
                    f'Shape/dtype changed: {key}')
            require(torch.equal(tensor.contiguous().view(torch.uint8),
                                original.contiguous().view(torch.uint8)),
                    f'Bits changed: {key}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved {destination}: 121 tensors, blocks 0..8; all tensor bytes verified.')


if __name__ == '__main__':
    main()
