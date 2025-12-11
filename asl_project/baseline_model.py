import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from omegaconf import OmegaConf


class BaselineCNN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        self.cfg = cfg

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        flat_size = 128 * 25 * 25

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.data.num_classes)
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=cfg.data.num_classes, average="macro")

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model.lr)
        return optimizer

