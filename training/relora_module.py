import os
import torch
import torch.nn as nn
import lightning as L
from torch import optim
from typing import Dict, Any
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import shutil

class ReloraModule(L.LightningModule):
    """ Relora base class for inheritance. """
    def __init__(self,
                 model_class: nn.Module,
                 model_path: str,
                 lora_config: LoraConfig = None,
                 train_dataset: Any = None,
                 eval_dataset: Any = None,
                 batch_size: int = 1,
                 learning_rate: float = 3e-4,
                 num_workers: int = 1,
                 **kwargs):
        
        super().__init__()
        # Paths for save-state juggling
        checkpoint_folder = "relora_checkpoints"
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.model_class = model_class
        self.model_path = model_path
        self.save_path = f"./{checkpoint_folder}/{model_path}"

        # Training and optimization
        self.learning_rate = learning_rate

        # Datasets
        self.train_set = train_dataset
        self.eval_set = eval_dataset

        # Misc Arguements
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # LoRA settings
        self.config = lora_config
        self.merge_precision = torch.float16 # If your CPU can actually handle it, I strongly advise using float32.
        self.load_precision = torch.float16 # Peft will cast this to float16 on its own, anyways
        self.has_merged = False # A flag to ensure our very first model reload comes from the original source.

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage: str):
        """Called before trainer.fit() and trainer.eval()"""
        if stage == 'fit':
            self.load_model(self.model_path)
            if self.config is not None: # We use this as our ReLoRA flag.
                self.init_lora()
    
    def configure_optimizers(self):
        """ Return whatever optimizers and learning rate schedulers you want here. At least one optimizer is required. """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        """Reinitialization logic"""
        self.train_set.dataset.initialize_dataset()
        return self.train_set

    def val_dataloader(self):
        self.eval_set.dataset.initialize_dataset()
        return self.eval_set

    def forward(self, x):
        return self.model(x)

    # ---------------------
    # RELORA METHODS
    # ---------------------
    def merge_lora_weights(self):
        """ Merges lora weights. Requires the current 0.6.0dev peft branch to work properly. """
        self.model = self.model.merge_and_unload() # merge lora

    def load_model(self, path):
        self.model = self.model_class.from_pretrained(path, load_in_8bit=True if self.trainer.precision == "int8" else False, torch_dtype=self.load_precision)

    def init_lora(self):
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.config)
        for name, param in self.model.named_parameters(): # Non-layer weights need to have training turned on.
            if any(substring in name for substring in ["norm", "bias", "lora_"]):
                param.requires_grad = True

    def reset_optimizer(self, in_place=False):
        """ The ReLoRA optimizer reset method. """
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    param_state = optimizer.state[p]
                    if in_place is False:
                        param_state["exp_avg"] = torch.zeros_like(p.data)
                        param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        param_state["exp_avg"].zero_()
                        param_state["exp_avg_sq"].zero_()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handles checkpoint logic"""
        if self.config is not None: # This is our flag to see if we're using the ReLoRA training method.
            self.reset_optimizer()
            self.merge_lora_weights
            checkpoint["state_dict"] = self.model.state_dict()  #We want to make sure we're not saving the LoRA state dict
            self.reset_optimizer(in_place=True) if self.trainer.scaler is not None else self.reset_optimizer()    
            self.init_lora()
        return checkpoint
