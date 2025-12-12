import logging
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from asl_project.callbacks import ComprehensivePlotsCallback
from asl_project.data import ASLDataModule
from asl_project.export import export_to_onnx, load_model_from_checkpoint
from asl_project.utils import current_git_commit

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    root_dir = Path(to_absolute_path("."))
    checkpoint_dir = root_dir / "models/checkpoints"
    plots_dir = root_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg.callbacks.model_checkpoint.dirpath = str(checkpoint_dir)

    dm = ASLDataModule(cfg)
    dm.setup()

    num_classes = len(dm.class_names)
    logger.info(f"Detected {num_classes} classes.")
    if cfg.model.get("num_classes", -1) != num_classes:
        OmegaConf.update(cfg, "model.num_classes", num_classes)

    model = instantiate(cfg.model)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        log_model=True,
        run_name=f"{cfg.model.name}-v{current_git_commit()[:7]}",
    )

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.callbacks.early_stopping)

    plots_callback = ComprehensivePlotsCallback(
        save_dir=str(plots_dir), class_names=dm.class_names
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback, plots_callback],
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        logger.info(f"Best checkpoint path: {best_ckpt_path}")

        final_ckpt_path = checkpoint_dir / "best.ckpt"
        shutil.copy(best_ckpt_path, final_ckpt_path)
        logger.info(f"Copied best model to {final_ckpt_path}")

        if cfg.get("auto_export_onnx", True):
            logger.info("Auto-exporting to ONNX...")
            export_model = load_model_from_checkpoint(str(final_ckpt_path), cfg.model)
            onnx_path = root_dir / "models/model.onnx"

            try:
                export_to_onnx(export_model, str(onnx_path), tuple(cfg.data.img_size))
                mlflow_logger.experiment.log_artifact(
                    mlflow_logger.run_id, str(onnx_path)
                )
            except Exception as e:
                logger.error(f"ONNX Export failed: {e}")

    logger.info(f"Plots saved in: {plots_dir}")

    if cfg.data.test_split > 0:
        trainer.test(model, datamodule=dm, ckpt_path="best")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
