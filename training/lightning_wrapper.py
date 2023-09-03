import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from typing import Dict, Any
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from peft import LoraModel, LoraConfig, get_peft_model

"""
This all seems to work as expected. Going to try larger scale testing
To-do: CLI implementation.
"""

#Easy lookup table for llama's LoRA modules
lora_modules = {"basic": ["q_proj", "v_proj"], "full": ["q_proj", "k_proj", "v_proj", "o_proj", "wte", "lm_head"], None: ""}

class ReloraModule(L.LightningModule):
    """Relora model wrapper. Currently works with classification tasks on a batch_size of 1."""
    def __init__(self,
                 model: nn.Module,
                 lora_config: LoraConfig = None,
                 train_dataset: Any = None,
                 eval_dataset: Any = None,
                 batch_size: int = 1,
                 learning_rate: float = 3e-4,
                 num_workers: int = 1,
                 **kwargs):
        
        super().__init__()
        # Training and optimization
        self.model = model
        self.config = lora_config

        # Datasets
        self.train_set = train_dataset
        self.eval_set = eval_dataset

        # Misc Arguements
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def setup(self, stage: str):
        """Called before trainer.fit() and trainer.eval()"""
        if stage == 'fit' and self.config is not None:
            self.init_lora()
            #May need to move prepare_model_for_8bit_training here, too.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop with the data from the training dataloader passed in as `batch`. """
        output = self(batch)
        loss = output["loss"]

        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, input_dc, batch_idx):
        """Lightning calls this inside the validation loop with the data from the validation dataloader passed in as `batch`."""
        output = self(batch)
        val_loss = output["loss"]

        self.log("val_loss", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_set

    def val_dataloader(self):
        return self.eval_set

    # ---------------------
    # RELORA METHODS
    # ---------------------  
    def merge_lora_weights(self):
        self.model.merge_and_unload()

    def init_lora(self):
        self.model = get_peft_model(self.model, self.config)

    def reset_optimizer(self):
        """Implements only the """

        for group in self.trainer.optimizers[0].param_groups:
            for p in group["params"]:
                param_state = self.trainer.optimizers[0].state[p]
                param_state["exp_avg"] = torch.zeros_like(p.data)
                param_state["exp_avg_sq"] = torch.zeros_like(p.data)
        #Of note, ReLoRA has two "settings", one just saves and loads the optimizer again, instead of setting all values to 0.
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handles checkpoint logic"""

        self.merge_lora_weights()
        #We want to make sure we're not saving the LoRA state dict
        checkpoint['state_dict'] = self.model.state_dict() 
        self.reset_optimizer()
        self.init_lora()
        return checkpoint
