import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from transformers import LlamaForCausalLM
from datasets import load_dataset
from peft import LoraConfig

from training.plugins import Int8Precision
from training.training_modules import ReloraModuleForLM

# ---------------
# Training Params
# ---------------
project_name = "lightning_relora_tests"
train_precision = "int8" #["32", "bf16-mixed", "16-mixed", "int8"]
dev_mode = False
use_wandb = False
max_epochs = 10
max_steps = 1.0 # Passing a float is interpreted as a percent (eg 100% here). Passing an int is interpreted as an integer number of batches.
max_val_steps = 1.0
# ---------------

# Library variables
L.seed_everything(234)
torch.set_float32_matmul_precision("medium")

def main(args):
    """ Integrating BYOM w/ CLI might be tough. """
    # --- Initialize datasets ---
    train_set = load_dataset("vicgalle/alpaca-gpt4", split="train")
    train_set = DataLoader(train_set, batch_size=1)
    val_set = load_dataset("vicgalle/alpaca-gpt4", split="eval")
    val_set = DataLoader(val_set, batch_size=1)
    
    # --- Initialize modules ---
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", inference_mode=False, modules_to_save="RMSNorm",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "wte", "lm_head"])
    model = ReloraModuleForLM(model_class=LlamaForCausalLM, model_path="meta-llama/Llama-2-7b", lora_config=lora_config,
                             train_dataset=train_set, eval_dataset=val_set, learning_rate=3e-4)
    
    # --- Automatically nitialize training variables based on the training params above ---
    precision = "32-true"
    plugins = []
    if train_precision == "int8":
        plugins.append(Int8Precision())
    else:
        precision = train_precision
    logger = WandbLogger(project=project_name) if use_wandb is True and dev_mode is False else None
    train_steps = max_steps if dev_mode is False else 10
    val_steps = max_val_steps if dev_mode is False else 5
    
    # --- Run ---
    trainer = L.Trainer(max_epochs=max_epochs, precision=precision, logger=logger, val_check_interval=1.0,
                        plugins=plugins, limit_train_batches=train_steps, limit_val_batches=val_steps)
    trainer.fit(model)

if __name__ == '__main__':
    main()
