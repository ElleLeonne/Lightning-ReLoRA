import torch
import os
from argparse import ArgumentParser
from transformers import LlamaForCausalLM
from peft import LoraConfig

from datasets import load_dataset, Dataset
import lightning as L
from training.lightning_wrapper import ReloraModule, lora_modules

# Library variables
L.seed_everything(234)
torch.set_float32_matmul_precision("medium")

def main(args):
    """ We need to find a way to integrate this with Lightning's CLI object, ideally. """
    # 1: Init datasets
    train_set = load_dataset("vicgalle/alpaca-gpt4", split="train")
    train_set = DataLoader(train_set, batch_size=1)
    val_set = load_dataset("vicgalle/alpaca-gpt4", split="eval")
    val_set = DataLoader(val_set, batch_size=1)
    
    # 2: Init model
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", inference_mode=False, modules_to_save="RMSNorm",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "wte", "lm_head"])

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b",
                                             load_in_8bit=True,)
    model = ReloraModule(model=model,
                         lora_config=lora_config,
                         train_set = train_set,
                         eval_set = val_set,)

    # 3: Begin training
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model)

if __name__ == '__main__':
    main()
