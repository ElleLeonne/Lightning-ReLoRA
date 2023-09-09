import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from transformers import ViTForImageClassification
from datasets import load_dataset
from peft import LoraConfig
from training.plugins import Int8Precision
from training.custom_datasets import BasicImageDataset
from training.training_modules import ReloraModuleForClassification

# --------------------
# Project Parameters
# --------------------
project_name = "image_classification"
train_precision = "bf16-mixed" #["32", "bf16-mixed", "16-mixed", "int8"]
dev_mode = False
use_wandb = False
use_lora = True
dataset_shards = 100
# -- trainer params --
max_epochs = 10
lora_merge_epochs = 1
learning_rate = 3e-4
max_steps = 1e5
max_val_steps = 1e4
# ---------------

# Library variables
L.seed_everything(234)
torch.set_float32_matmul_precision("medium")

def main():
    # Dataset preperations
    train_set = BasicImageDataset(load_dataset(path="imagenet-1k", split="train"), length=1281167, shards=dataset_shards)
    val_set = BasicImageDataset(load_dataset(path="imagenet-1k", split="validation"), length=100000, shards=dataset_shards//10)

    # Model init
    model_class = ViTForImageClassification
    model_path = "google/vit-base-patch16-224"
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", inference_mode=False)
    # If you choose not supply target_modules, the module will automatically apply lora to every possible compatible layer.

    # ------------------------------
    # Auto initialize trainer params
    # ------------------------------
    precision = "32-true"
    plugins = []
    if train_precision in ["int-8", "int8"]:
        plugins.append(Int8Precision())
    else:
        precision = train_precision
    logger = WandbLogger(project=project_name) if use_wandb is True and dev_mode is False else None
    train_steps = max_steps if dev_mode is False else 10
    val_steps = max_val_steps if dev_mode is False else 5
    epochs = max_epochs if dev_mode is False else 5
    lora_config = lora_config if use_lora is True else None
    epoch_merge_freq = lora_merge_epochs if use_lora is True else 0

    # --- Run ---
    model = ReloraModuleForClassification(model_class=model_class, model_path=model_path, lora_config=lora_config, lora_merge_freq=epoch_merge_freq,
                                          train_dataset=train_set, eval_dataset=val_set, learning_rate=learning_rate)
    trainer = L.Trainer(max_epochs=epochs, precision=precision,logger=logger, val_check_interval=1.0, log_every_n_steps=5000,
                        limit_train_batches=train_steps, limit_val_batches=val_steps, plugins=plugins, reload_dataloaders_every_n_epochs=1)
    trainer.fit(model)

if __name__ == "__main__":
    main()