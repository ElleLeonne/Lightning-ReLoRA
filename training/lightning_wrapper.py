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
from accelerate import Accelerator

#Easy lookup table for llama's LoRA modules
lora_modules = {"basic": ["q_proj", "v_proj"], "full": ["q_proj", "k_proj", "v_proj", "o_proj", "wte", "lm_head"], None: ""}

class ReloraModule(LightningModule):
    """
    Sample model to show how to define a template.

    Example:

        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     drop_prob=0.2,
        ...     batch_size=2,
        ...     in_features=28 * 28,
        ...     learning_rate=0.001 * 8,
        ...     optimizer_name='adam',
        ...     data_root='./datasets',
        ...     out_features=10,
        ...     num_workers=4,
        ...     hidden_dim=1000,
        ... )
        >>> model = LightningTemplateModel(**params)
    """

    def __init__(self,
                 model: nn.Module,
                 lora_config: LoraConfig = None,
                 accelerator: Accelerator = Accelerator(),
                 train_dataset: Any = None,
                 eval_dataset: Any = None,
                 test_dataset: Any = None,
                 drop_prob: float = 0.2,
                 batch_size: int = 2,
                 learning_rate: float = 0.001 * 8,
                 optimizer_name: str = 'adamw',
                 num_workers: int = 4,
                 **kwargs):
        
        super().__init__()
        # Training and optimization
        self.model = model,
        self.config = lora_config,
        self.accelerator = accelerator
        #The trainer carries its own accelerator, but we don't have access to it.

        # Datasets
        self.train_set = train_dataset
        self.eval_set = eval_dataset
        self.test_set = test_dataset

        # Misc Arguements
        self.num_workers = num_workers
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

    def setup(self, stage: str):
        """Called before trainer.fit() and trainer.eval()"""
        if stage == 'fit' and self.lora_config is not None:
            self.init_lora()
            #May need to move prepare_model_for_8bit_training here, too.

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """ Lightning calls this inside the training loop with the data from the training dataloader passed in as `batch`. """

        input_ids, labels = batch
        logits = self(input_ids)

        """ # For classification tasks, you'll want to use an accuracy metric
        predictions = torch.argmax(logits[-1, :), dim=-1)
        acc = self.accuracy(preds, labels)
        self.log("train_accuracy", acc, on_step=True, on_epoch=False, prog_bar=True)
        loss = F.cross_entropy(predictions, labels)"""
        
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, batch_size=1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, labels = batch
        logits = self(input_ids)
        val_loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        n_correct_preds = torch.sum(y == preditions).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_preds, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        test_loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        n_correct_preds = torch.sum(y == preds).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_preds, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        self.log('val_loss', avg_loss)
        self.log('val_acc', val_acc)
        return {'val_loss': avg_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Preprocessing"""
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    #Lightning 2.x has a LightningCLI object now, will need to investigate.
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """ Define parameters that only apply to this model """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # LoRA params
        parser.add_argument('--lora_r', default=8, type=int) #LoRA rank, How deep our LoRA layers are
        parser.add_argument('--lora_alpha', default=16, type=int) #The amount of compression used, aim for roughly double the rank
        parser.add_argument('--lora_dropout', default=0.05, type=float)
        parser.add_argument('--lora_modules', default="basic", choices=['basic', 'full'])


        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--num_workers', default=4, type=int)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser

    # ---------------------
    # RELORA METHODS
    # ---------------------
    def set_lora_config(self, **kwargs):
        """Basic method to set LoRA config from outside instance initiation."""
        self.config = LoraConfig(**kwargs)
        return self.config
    
    def merge_lora_weights(self):
        self.model.merge_and_unload()

    def init_lora(self):
        self.model = get_peft_model(self.model, self.config)

    def reset_optimizer(self, optimizer):
        """This was part of the trainer originally, references may need to be changed, since we now only has have
        access to the wrapper and the initial training args"""
        self.log.info("Resetting optimizer states to zeros")

        #This assumes that optimizer is not wrapped by accelerator
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = optimizer.state[p]
                param_state["exp_avg"] = torch.zeros_like(p.data)
                param_state["exp_avg_sq"] = torch.zeros_like(p.data)
        #Of note, ReLoRA has two "settings", one just saves and loads the optimizer again, instead of setting all values to 0.
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handles checkpoint logic"""

        #Merging and reinitializing LoRA
        self.merge_lora_weights()
        checkpoint['state_dict'] = self.model.state_dict() #We want to make sure we're not saving the LoRA state dict
        if self.accelerator is not None: #Unwrap optimizer
            self.optimizer = self.optimizer.optimizer
        self.reset_optimizer(self.optimizer)
        self.init_lora()
        if self.accelerator is not None:
            self.model = self.accelerator.prepare_model(self.model)
            self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)

        output = super().on_save_checkpoint(checkpoint)
        return output
