from pathlib import Path

import hydra
import torch
import torch.onnx
from omegaconf import DictConfig

from asl_project.model import ASLModule


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def convert(cfg: DictConfig):
    if not Path(cfg.checkpoint_path).exists():
        print(f"Checkpoint not found: {cfg.checkpoint_path}")
        return

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    model = ASLModule.load_from_checkpoint(cfg.checkpoint_path, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.img_size[0], cfg.img_size[1])
    output_path = cfg.onnx_path

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("ONNX export complete.")

    # TensorRT Hint (since we are in Python environment without trt installed usually)
    print("\nTo convert to TensorRT, run the following command (requires trtexec):")
    # Using trtexec CLI
    # print(
    #     f"trtexec --onnx={output_path}
    # )


if __name__ == "__main__":
    convert()
