import yaml

transforms = []

for i in range(16):
    p = f"gpt_neox.layers.{i}.attention"
    # query_key_value.weight: rows 0..3839, 4608..6143
    transforms.append({
        "concat": {
            "from": [f"{p}.query_key_value.weight::[:3840, :]",
                     f"{p}.query_key_value.weight::[4608:, :]"],
            "to": f"{p}.query_key_value.weight_pruned",
            "dim": 0,
        }
    })
    # query_key_value.bias: rows 0..3839, 4608..6143
    transforms.append({
        "concat": {
            "from": [f"{p}.query_key_value.bias::[:3840]",
                     f"{p}.query_key_value.bias::[4608:]"],
            "to": f"{p}.query_key_value.bias_pruned",
            "dim": 0,
        }
    })
    # dense.weight: columns 0..1279, 1536..2047
    transforms.append({
        "concat": {
            "from": [f"{p}.dense.weight::[:, :1280]",
                     f"{p}.dense.weight::[:, 1536:]"],
            "to": f"{p}.dense.weight_pruned",
            "dim": 1,
        }
    })

# delete originals and rename pruned into place
for i in range(16):
    p = f"gpt_neox.layers.{i}.attention"
    transforms.append({"delete": {"target": f"{p}.query_key_value.weight"}})
    transforms.append({"delete": {"target": f"{p}.query_key_value.bias"}})
    transforms.append({"delete": {"target": f"{p}.dense.weight"}})
    transforms.append({"move": {"from": f"{p}.query_key_value.weight_pruned", "to": f"{p}.query_key_value.weight"}})
    transforms.append({"move": {"from": f"{p}.query_key_value.bias_pruned", "to": f"{p}.query_key_value.bias"}})
    transforms.append({"move": {"from": f"{p}.dense.weight_pruned", "to": f"{p}.dense.weight"}})

# required checks
transforms.append({"assert": {"shape": {"of": "gpt_neox.layers.0.attention.query_key_value.weight", "is": [5376, 2048]}}})
transforms.append({"assert": {"shape": {"of": "gpt_neox.layers.0.attention.query_key_value.bias", "is": [5376]}}})
transforms.append({"assert": {"shape": {"of": "gpt_neox.layers.0.attention.dense.weight", "is": [2048, 1792]}}})
transforms.append({"assert": {"count": {"of": ".*", "is": 244}}})

plan = {
    "inputs": ["model::inputs/base/model.safetensors"],
    "transforms": transforms,
    "output": {
        "path": "out/T2/model.safetensors",
        "format": "safetensors",
    },
}

with open("out/T2/plan.yaml", "w") as f:
    yaml.dump(plan, f, sort_keys=False, width=100)
