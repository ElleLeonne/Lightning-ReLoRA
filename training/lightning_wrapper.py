"""
Basic lightning module wrapper. Training loop features go here.
Lifted from -> https://github.com/ibeltagy/pytorch-lightning/blob/master/pl_examples/models/lightning_template.py
"""
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from typing import Dict, Any
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from peft import LoraModel, LoraConfig, get_peft_model

class LightningTemplateModel(LightningModule):
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
                 lora_config: LoraConfig,
                 drop_prob: float = 0.2,
                 batch_size: int = 2,
                 learning_rate: float = 0.001 * 8,
                 optimizer_name: str = 'adam',
                 data_root: str = './datasets',
                 num_workers: int = 4,
                 **kwargs
                 ):

        # init superclass
        super().__init__()
        self.model = model,
        self.config = lora_config,
        self.num_workers = num_workers
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.data_root = data_root
        
        self.example_input_array = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

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
        MNIST(self.data_root, train=True, download=True, transform=transforms.ToTensor())
        MNIST(self.data_root, train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        self.mnist_train = MNIST(self.data_root, train=True, download=False, transform=transform)
        self.mnist_test = MNIST(self.data_root, train=False, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--num_workers', default=4, type=int)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser

    # ---------------------
    # RELORA METHODS
    # ---------------------
    def merge_lora_weights(self):
        self.model.merge_and_unload()

    def reinit_lora(self):
        self.model = get_peft_model(self.model, self.config)

    def reset_optimizer(self, optimizer):
        """This was part of the trainer originally, references may need to be changed, since we now only has have
        access to the wrapper and the initial training args"""
        self.fabric.logger.info("Resetting optimizer states to zeros")

        #This assumes that optimizer is not wrapped by accelerator
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = optimizer.state[p]
                param_state["exp_avg"] = torch.zeros_like(p.data)
                param_state["exp_avg_sq"] = torch.zeros_like(p.data)
        #Of note, ReLoRA has two "settings", one just saves and loads the optimizer again, instead of setting all values to 0.
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_save_checkpoint(checkpoint)
