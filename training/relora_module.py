import sys
import math
import torch
import torch.nn as nn
import lightning as L
from torch import optim
import torch.utils.data as td
from typing import Any
from .utils import Accuracy

import torch._dynamo # Auto-resolve torch.compile errors.
torch._dynamo.config.suppress_errors = True

peft_supported_layers = [""]

class LightningModel(L.LightningModule):
    """ Relora base class for inheritance. """
    def __init__(self, model_class: nn.Module, model_path: str = None, collator = None, lora_merge_freq: int = 0, learning_rate: float = 3e-4,
                 train_dataset: Any = None, eval_dataset: Any = None, test_dataset: Any = None, batch_size: int = 1, num_workers: int = 4, **kwargs):
        """ Args:

            - model_class: Either a preloaded model object, or a base class that you intend to train with. Using a full object limits precision tricks.
            - model_path: The path to load the model from if it's a huggingface model. 'None' assumes model_class is a pre-loaded model object.
            - lora_config: The configuration file to use when re-initializing LoRA. 'None' disables ReLoRA and runs normal training. 
            - lora_merge_freq: How many epochs are required before merging and resetting the LoRA layers into the base model. 
            - learning_rate: How quickly the weights update. LoRA is very sensitive to learning rate, and the layers overfit quite quickly.
            - train/eval/test_dataset: Dataset objects that you intend to train on. Override self.collator in subclasses for custom batching behavior.
            - batch_size: How many samples per batch for the dataloaders to use. Setting this higher speeds up training, but uses more memory.
            - num_workers: How many workers the Dataloader should launch to preprocess your datasets, larger batch sizes require more workers.
            """
        super().__init__(**kwargs)
        # Paths for save-state juggling
        self.model_class = model_class # The trainer will understand if you pass an instantiated model
        self.model_path = model_path
        self.collator = collator # If your model uses a custom batching function, you'll want to set this in your subclass.
        self.accuracy = Accuracy()

        # Training and optimization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.workers = num_workers

        # Datasets
        self.train_set = train_dataset
        self.eval_set = eval_dataset
        self.test_set = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
 
        # LoRA settings
        self.torch_compile = True # You can set this flag to False if torch.dynamo fails for you, to avoid using pytorch 2.0's compile method.
        self.verbose = True
    # ---------------------
    # TRAINING LOOP
    # ---------------------
    def forward(self, x):
        return self.model(x)

    def training_step(self, input_dc, batch_idx):
        print("Please implement the 'training_step' method in a subclass of this parent class.")
        return None
    def validation_step(self, input_dc, batch_idx):
        print("Please implement the 'validation_step' method in a subclass of this parent class.")
        return None

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def setup(self, stage: str):
        """Called before trainer.fit() and trainer.eval()"""
        if stage == 'fit':
            self.load_model(self.model_path)
            if torch.__version__ <= "2":
                print("Pytorch's compiler is not compatible with versions lower than 2.0 - Performing standard training.")
            elif sys.platform == "win32":
                print("Pytorch's compiler is only compatible with 64 bit versions of Windows. - Performing standard training.")
            elif sys.version_info.major == 3 and sys.version_info.minor >= 12:
                print("Pytorch's compiler is not yet compatible with Python " + str(sys.version_info.major)+"."+str(sys.version_info.minor) + " - Performing standard training.")
            elif self.torch_compile is False:
                print("torch.compile has been manually disabled. - Performing standard training.")
            else:
                self.model = torch.compile(self.model)
            self.model = self.model.train() # Huggingface explicitly requires this now.

        if stage == 'eval':
            self.model = self.model.eval()

    def configure_optimizers(self):
        """ Return whatever optimizers and learning rate schedulers you want here. At least one optimizer is required. """
        print("Preparing optimizers")
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_global_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self): # Lightning seems to not be prepared for "eval only" runs, and expects a training set.
        print("Preparing training dataloader") 
        return td.DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collator, num_workers=self.workers) if self.exists("train_set") else None
    def val_dataloader(self): # Lightning also seems to not like runs without an eval set. Another one for the 'todo' list.
        print("Preparing eval dataloader")
        return td.DataLoader(self.eval_set, batch_size=self.batch_size, collate_fn=self.collator, num_workers=self.workers) if self.exists("eval_set") else None
    def test_dataloader(self):
        print("Preparing test dataloader")
        return td.DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collator, num_workers=self.workers) if self.exists("test_set") else None
    def predict_dataloader(self): # This is new, another thing for the todo list. Seems to be a way to optimize for only specific responses.
        print("Predict dataloader not implemented yet.")
    # ---------------------
    # RELORA METHODS
    # ---------------------
    def merge_lora_weights(self):
        """ Merges lora weights. Requires the current 0.6.0dev peft branch to work properly. """
        self.model = self.model.merge_and_unload() # merge lora

    def load_model(self, path):
        self.model = (self.model_class if self.model_path is None else
                      self.model_class.from_pretrained(path, load_in_8bit=True if self.trainer.precision == "int8" else False, torch_dtype=self.load_precision))

    def on_train_epoch_end(self):
        if self.exists("accuracy"):
            self.accuracy.reset()

    # ---------------------
    # Helper Methods
    # ---------------------
    def exists(self, attribute):
        """ Checks the truthiness of an attribute, good for unit-testing, stacking custom inheritance classes, and conditionals. """
        return bool(hasattr(self, attribute) and bool(getattr(self, attribute)))
    # We need the number of steps for our optimizer ASAP, but Lightning prefers to calculate this stuff dynamically, and explicit variables are easier to grok.
    @property
    def max_global_steps(self): # Max steps rarely carries the proper total step size, so this calculates it manually while I try to track down the trainer's own reference.
        """ Calculates the total step count required to complete the entire training session over all epochs. """
        return math.ceil(self.num_batches / self.num_devices) * self.num_epochs # Progress bar only runs along rank 0, so we only need to round up.
    @property
    def num_epochs(self):
        """ Fetches the total number of epochs that we'll be training for """
        return self.trainer.max_epochs if self.trainer.max_epochs != -1 else None
    @property
    def num_batches(self):
        """ Fetches the total number of batches/steps that the trainer wants to use for a single training pass. """
        return self.trainer.num_training_batches if self.trainer.num_training_batches not in [-1, math.inf] else math.ceil(self.train_set.__len__() / self.batch_size)
    @property
    def num_devices(self):
        """ Fetches the total number of devices that the trainer will split the data between. """
        devices = (self.trainer.num_nodes * self.trainer.num_devices)
        return devices if devices != -1 else 1
