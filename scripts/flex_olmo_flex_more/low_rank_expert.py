import logging
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import typer

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

def main(
    model_path: str = typer.Argument(..., help="Path to the FlexOLMo model in HF format"),
    rank: list[int] = typer.Option(
        [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384],
        help="Rank for the low-rank adapters to be applied to each linear layer",
    ),
    processes: int = typer.Option(
        1,
        help="Number of processes for SVD computation",
    ),
    lora_modules: list[str] = typer.Option(
        [
            "gate_proj.weight",
            "down_proj.weight",
            "up_proj.weight",
        ],
        help="List of modules to apply LoRA to",
    )
):
    prepare_cli_environment()
    log.info(f"Setting number of threads for SVD computation to {processes}")
    torch.set_num_threads(processes)

    log.info(f"Loading config from {model_path}")
    model_config = AutoConfig.from_pretrained(model_path)
    log.info(model_config)

    log.info("Initializing empty model")
    model = AutoModelForCausalLM.from_config(model_config)
    model_state_dict = model.state_dict()

    log.info(f"Loading model from {model_path}")
    expert = AutoModelForCausalLM.from_pretrained(model_path)
    expert_state_dict = expert.state_dict()

    key2usvh = {}
    for r in rank:
        for key in list(model_state_dict.keys()):
            weights = expert_state_dict[key]
            if ".experts." in key:
                log.info(f"Processing key {key}")
                expert = int(key.split(".experts.")[1].split(".")[0])
                assert expert in (0, 1), f"Expert can only be 0 or 1: {key}"
                print(f"Expert: {expert}")
                if expert and any(lora_module in key for lora_module in lora_modules):
                    base_expert = expert_state_dict[key.replace(".experts.1.", ".experts.0.")]
                    delta_expert = weights - base_expert
                    # compute the low-rank adaptation
                    if key not in key2usvh:
                        log.info(f"Computing SVD for key {key}")
                        key2usvh[key] = torch.linalg.svd(delta_expert, full_matrices=False)
                    u, s, vh = key2usvh[key]
                    lora_u = u[:, :r]
                    lora_s = s[:r]
                    lora_vh = vh[:r, :]
                    # reconstruct the low-rank expert adaptation                
                    lora_expert = (lora_u * lora_s) @ lora_vh
                    weights = lora_expert + base_expert
            model_state_dict[key] = weights
        # save the final_state_dict for the MoE in a format that the olmo_core trainer likes
        save_path = f"{model_path}-r{r}"
        log.info(f"Saving model to {save_path}")
        model.save_pretrained(save_path, state_dict=model_state_dict)
    log.info("Done")

if __name__ == "__main__":
    typer.run(main)
