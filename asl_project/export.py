import logging
from pathlib import Path
from typing import Tuple

import hydra
import torch
import torch.onnx
from hydra.utils import get_class, to_absolute_path
from omegaconf import DictConfig

from asl_project.utils import ensure_dir

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, model_cfg: DictConfig):
    model_cls = get_class(model_cfg._target_)
    model = model_cls.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.freeze()
    return model


def export_to_onnx(
    model: torch.nn.Module, output_path: str, img_size: Tuple[int, int]
) -> None:
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])

    logger.info(f"Exporting ONNX to {out_path} with opset 17...")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"ONNX model successfully exported to {out_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def convert(cfg: DictConfig):
    checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    output_path = to_absolute_path(cfg.onnx_path)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(str(checkpoint_path), cfg.model)
    img_size = tuple(cfg.img_size)

    export_to_onnx(model, str(output_path), img_size)


if __name__ == "__main__":
    convert()
