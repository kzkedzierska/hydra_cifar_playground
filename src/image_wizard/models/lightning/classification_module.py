import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

__all__ = ["ClassificationModule"]


class ClassificationModule(L.LightningModule):
    """Lightning module for CIFAR-10 classification."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        scheduler_name: str | None = None,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.optimizer_kwargs = optimizer_kwargs

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        self.train_acc(logits, y)

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        self.val_acc(logits, y)

        # Logging
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        self.test_acc(logits, y)

        # Logging
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)

        return loss

    def configure_optimizers(self):
        # Get optimizer class
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)

        config = {"optimizer": optimizer}

        # Add scheduler if specified
        if self.scheduler_name:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_name)
            scheduler = scheduler_class(optimizer)
            config["lr_scheduler"] = {"scheduler": scheduler, "monitor": "val/loss"}

        return config
