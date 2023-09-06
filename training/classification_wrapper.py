import torch
import torch.nn as nn
import lightning as L
from torch import optim
from typing import Dict, Any
from peft import PeftModel, LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss

class ReloraModule(L.LightningModule):
    """Relora model wrapper. Currently works with classification tasks on a batch_size of 1."""
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
        # Training and optimization
        self.model_class = model_class
        self.model_path = model_path
        self.save_path = f"{model_path}_checkpoint"
        
        self.config = lora_config
        self.loss = CrossEntropyLoss()

        # Datasets
        self.train_set = train_dataset
        self.eval_set = eval_dataset

        # Misc Arguements
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # LoRA settings
        self.merge_precision = torch.float32 # If your CPU can actually handle the model size, I strongly advise using float32.
        self.load_precision = torch.float16 # Peft will cast this to float16 on its own, anyways

    def setup(self, stage: str):
        """Called before trainer.fit() and trainer.eval()"""
        if stage == 'fit' and self.config is not None:
            self.load_model(self.model_path)
            self.init_lora()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """ Lightning calls this inside the training loop with the data from the training dataloader passed in as `batch`. """
        output = self(batch)
        logits = output["hidden_state"][:, -1, :]
        labels = output["labels"]

        loss = self.loss(logits, torch.tensor(labels, device=logits.device))
        preds = torch.argmax(logits, dim=-1)

        accuracy = (preds == labels).float().sum() / len(labels)
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, batch_size=self.batch_size, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Lightning calls this inside the validation loop with the data from the validation dataloader passed in as `batch`. """
        output = self(batch)
        logits = output["hidden_state"][:, -1, :]
        labels = output["labels"]

        val_loss = self.loss(logits, torch.tensor(labels, device=logits.device))
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels).float().sum() / len(labels)
        
        self.log("val_loss", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return val_loss, accuracy

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """ Return whatever optimizers and learning rate schedulers you want here. At least one optimizer is required. """
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
    def merge_lora_weights_8bit(self):
        """We can't currently merge lora weights while the model is loaded in 8bit mode, so we have to dance a little."""
        lora_path = f"{self.model_path}_lora"
        self.model.save_pretrained(lora_path) # Save lora weights
        del self.model # We do this outright to avoid transitional memory spaces. Merging requires a LOT of memory, so we favor the CPU.
        model = self.model_class.from_pretrained(self.model_path, device_map={"": "cpu"}, torch_dtype=self.merge_precision) # Load in higher precision if able

        #first_weight = model.model.layers[0].self_attn.q_proj.weight #Llama has nested model attributes.
        og_weight = model.classification_head.layers[0].self_attn.q_proj.weight

        model = PeftModel.from_pretrained(model, lora_path, device_map={"": "cpu"}, torch_dtype=self.merge_precision)

        self.model = model.merge_and_unload() # Load and merge lora

        new_weight = self.model.classification_head.layers[0].self_attn.q_proj.weight
        assert torch.allclose(og_weight, new_weight) # Check to ensure the weights actually changed.
        del og_weight, new_weight, model # Deleting them, because they're big.

    def merge_lora_weights(self):
        """ We should strip the merge check prior to full release, to ensure modularity. """
        og_weight = self.classification_head.layers[0].self_attn.q_proj.weight
        self.model = self.model.to(dtype=self.merge_precision)
        self.model = self.model.merge_and_unload() # merge lora
        self.model = self.model.to(dtype=self.load_precision)
        new_weight = self.model.classification_head.layers[0].self_attn.q_proj.weight
        assert torch.allclose(og_weight, new_weight) # Check to ensure the weights actually changed.
        del og_weight, new_weight # Deleting them, because they're big.

    def load_model(self, path):
        self.model = self.model_class.from_pretrained(path, load_in_8bit=True, torch_dtype=self.load_precision)

    def init_lora(self):
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.config)

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

    def seed_scaler(self):
        """ In mixed precision training, messing with the model or the optimizer results in the scaler receiving an out-of-sequence .update() method.
        Until this is working properly, I highly recommend using either float32, bf16-mixed, or int8 training. Do not use 16-mixed. """
        device_list = [p.device for p in self.parameters()]
        self.trainer.scaler._per_optimizer_states[id(self.trainer.optimizers)]["found_inf_per_device"] = {device: torch.tensor([0.0]) for device in device_list}
        #print({opt: state["found_inf_per_device"] for opt, state in self.trainer.scaler._per_optimizer_states.items()})

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handles checkpoint logic"""
        self.reset_optimizer()
        self.merge_lora_weights_8bit() if self.trainer.precision == "int8" else self.merge_lora_weights()
        checkpoint['state_dict'] = self.model.state_dict()  #We want to make sure we're not saving the LoRA state dict
        self.reset_optimizer(in_place=True) if self.trainer.scaler is not None else self.reset_optimizer()            
        self.init_lora()
        return checkpoint
