import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ComprehensivePlotsCallback(Callback):
    def __init__(self, save_dir: str, class_names: list):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

        self.history = defaultdict(list)

        self.val_preds = []
        self.val_targets = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(metrics["train_loss"].item())
        if "train_acc" in metrics:
            self.history["train_acc"].append(metrics["train_acc"].item())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y = batch
        logits = pl_module(x)
        preds = torch.argmax(logits, dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.history["val_loss"].append(metrics["val_loss"].item())
        if "val_acc" in metrics:
            self.history["val_acc"].append(metrics["val_acc"].item())

    def on_fit_end(self, trainer, pl_module):
        self._plot_curves(trainer)
        self._plot_confusion_matrix(trainer)

    def _plot_curves(self, trainer):
        fig, ax = plt.subplots(figsize=(10, 6))

        min_len = min(len(self.history["train_loss"]), len(self.history["val_loss"]))

        ax.plot(self.history["train_loss"][:min_len], label="Train Loss", marker="o")
        try:
            ax.plot(self.history["val_loss"][:min_len], label="Val Loss", marker="s")
        except (KeyError, IndexError):
            pass

        ax.set_title("Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

        loss_path = self.save_dir / "loss_curve.png"
        fig.savefig(loss_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        min_len_acc = min(len(self.history["train_acc"]), len(self.history["val_acc"]))

        ax.plot(self.history["train_acc"][:min_len_acc], label="Train Acc", marker="o")
        try:
            ax.plot(self.history["val_acc"][:min_len_acc], label="Val Acc", marker="s")
        except (KeyError, IndexError):
            pass

        ax.set_title("Accuracy Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)

        acc_path = self.save_dir / "acc_curve.png"
        fig.savefig(acc_path)
        plt.close(fig)

        if trainer.logger:
            trainer.logger.experiment.log_artifact(
                trainer.logger.run_id, str(loss_path)
            )
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, str(acc_path))

    def _plot_confusion_matrix(self, trainer):
        if not self.val_targets:
            return

        conf_mat = confusion_matrix(self.val_targets, self.val_preds, normalize="true")

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            conf_mat,
            annot=False,
            fmt=".2f",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("Confusion Matrix (Normalized)")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        cm_path = self.save_dir / "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)

        if trainer.logger:
            trainer.logger.experiment.log_artifact(trainer.logger.run_id, str(cm_path))

        self.val_preds = []
        self.val_targets = []
