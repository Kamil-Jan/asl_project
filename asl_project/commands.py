import logging
from pathlib import Path
from typing import List

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from asl_project.export import convert
from asl_project.infer import predict_image, run_webcam
from asl_project.train import train

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ASLCLI:
    def _get_config(
        self, config_path: str, config_name: str, overrides: List[str]
    ) -> DictConfig:
        project_root = Path(__file__).parent.parent.resolve()
        abs_config_dir = project_root / config_path

        if not abs_config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {abs_config_dir}")

        with initialize_config_dir(config_dir=str(abs_config_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=overrides)
            OmegaConf.resolve(cfg)
            return cfg

    def train(self, config_name: str = "train", **kwargs):
        overrides = [f"{k}={v}" for k, v in kwargs.items()]

        logger.info(f"Starting training with overrides: {overrides}")
        cfg = self._get_config("configs", config_name, overrides)
        train(cfg)

    def export(self, config_name: str = "inference", **kwargs):
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = self._get_config("configs", config_name, overrides)
        convert(cfg)

    def infer(self, image_path: str = None, config_name: str = "inference", **kwargs):
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        if image_path:
            overrides.append(f"input.image_path={image_path}")

        cfg = self._get_config("configs", config_name, overrides)

        if cfg.input.image_path:
            predict_image(cfg)
        else:
            raise ValueError("Please provide --image_path argument.")

    def webcam(self, config_name: str = "inference", **kwargs):
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        cfg = self._get_config("configs", config_name, overrides)
        run_webcam(cfg)


def main():
    fire.Fire(ASLCLI)


if __name__ == "__main__":
    main()
