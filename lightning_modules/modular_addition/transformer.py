from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics import Accuracy

from configs.modular_addition.transformer import config
from models.transformer import TransformerClassifier


class TransformerLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(asdict(config.model))
        self.train_config = config.train

        self.model = TransformerClassifier(**asdict(config.model))

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.model.num_classes)

    def forward(self, x, lengths):
        # Boolean causal mask: True means masked (upper triangular)
        B, T = x.size(0), x.size(1)
        causal = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
        mask = causal.unsqueeze(0).expand(B, -1, -1)  # (B,T,T)

        return self.model(x, lengths, mask=mask)

    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)

        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = torch.clamp(lengths - 1, min=0)
        final_targets = targets[batch_indices, last_indices]

        loss = self.criterion(logits, final_targets)

        self.train_acc(logits, final_targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)

        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = torch.clamp(lengths - 1, min=0)
        final_targets = targets[batch_indices, last_indices]

        loss = self.criterion(logits, final_targets)

        self.val_acc(logits, final_targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)

        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        last_indices = torch.clamp(lengths - 1, min=0)
        final_targets = targets[batch_indices, last_indices]

        loss = self.criterion(logits, final_targets)

        self.test_acc(logits, final_targets)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.train_config.optimizer)
        return optimizer