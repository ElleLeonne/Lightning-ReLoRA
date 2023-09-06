import torch
from torch.nn import CrossEntropyLoss
from relora_module import ReloraModule

class ReloraModuleForLM(ReloraModule):
    """ Relora module for language modeling, or other tasks without defined accuracy metrics. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        output = self(batch)
        logits = output["logits"][:, -1, :]
        labels = output["labels"]

        loss = self.loss(logits, labels)
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        logits = output["logits"][:, -1, :]
        labels = output["labels"]

        val_loss = self.loss(logits, labels)
        self.log("val_loss", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return val_loss

class ReloraModuleForClassification(ReloraModule):
    """ Includes accuracy metric in the logger """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        output = self(batch)
        logits = output["logits"][:, -1, :]
        labels = output["labels"]

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        accuracy = (preds == labels).float().sum() / len(labels)
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, batch_size=self.batch_size, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        logits = output["logits"][:, -1, :]
        labels = output["labels"]

        val_loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        accuracy = (preds == labels).float().sum() / len(labels)
        self.log("val_loss", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)

        return val_loss, accuracy
