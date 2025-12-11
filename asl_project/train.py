import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from asl_project.data import ASLDataModule


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    dm = ASLDataModule(cfg)

    model = instantiate(cfg.model)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(**cfg.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.callbacks.early_stopping)

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
