from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics import Accuracy, F1Score


class BaseASLModule(pl.LightningModule, ABC):
    """
    Abstract Base Class for ASL models.
    Handles metrics, logging, and training loop, but delegates model definition.
    """

    def __init__(self, num_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Child classes must instantiate self.net
        self.net = None

    @abstractmethod
    def build_network(self) -> nn.Module:
        """Must be implemented by child classes."""
        pass

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class BaselineCNN(BaseASLModule):
    def __init__(
        self,
        num_classes: int = 29,
        lr: float = 0.001,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__(num_classes, lr)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = self.build_network()

    def build_network(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, self.hidden_dim),  # Assumes 224x224
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )


class TransferResNet(BaseASLModule):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 29,
        lr: float = 0.0003,
    ):
        super().__init__(num_classes, lr)
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.net = self.build_network()

    def build_network(self):
        # Dynamically load from torchvision (e.g., models.resnet18)
        backbone_fn = getattr(models, self.backbone_name)
        weights = "DEFAULT" if self.pretrained else None
        model = backbone_fn(weights=weights)

        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final FC layer (fc is specific to ResNet)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, self.num_classes)
        return model
