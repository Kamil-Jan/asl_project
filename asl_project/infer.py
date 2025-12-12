import logging
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import hydra
import numpy as np
import onnxruntime as ort
import torch
from albumentations.pytorch import ToTensorV2
from hydra.utils import get_class, to_absolute_path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = cfg.class_names
        self.img_h, self.img_w = cfg.img_size
        self.runtime = cfg.runtime

        self.transform = A.Compose(
            [
                A.Resize(self.img_h, self.img_w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        if self.runtime == "onnx":
            self._load_onnx(to_absolute_path(cfg.onnx_path))
        else:
            self._load_torch(to_absolute_path(cfg.checkpoint_path), cfg.model)

    def _load_onnx(self, model_path: str) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")

        logger.info("Loading ONNX session from %s", model_path)
        self.ort_session = ort.InferenceSession(
            model_path, providers=self.cfg.providers
        )

    def _load_torch(self, checkpoint_path: str, model_cfg: DictConfig) -> None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        model_cls = get_class(model_cfg._target_)
        logger.info("Loading Torch checkpoint from %s", checkpoint_path)
        self.model = model_cls.load_from_checkpoint(checkpoint_path).to(self.device)
        self.model.eval()
        self.model.freeze()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        augmented = self.transform(image=image)
        return augmented["image"].unsqueeze(0)

    def _predict_onnx(self, tensor: torch.Tensor) -> Tuple[int, float]:
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: tensor.numpy()}
        logits = self.ort_session.run(None, ort_inputs)[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1)
        prob, idx = torch.max(probs, dim=1)
        return idx.item(), prob.item()

    def _predict_torch(self, tensor: torch.Tensor) -> Tuple[int, float]:
        with torch.no_grad():
            tensor = tensor.to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            prob, idx = torch.max(probs, dim=1)
            return idx.item(), prob.item()

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        tensor = self.preprocess(image)

        if self.runtime == "onnx":
            idx, prob = self._predict_onnx(tensor)
        else:
            idx, prob = self._predict_torch(tensor)

        if prob < self.cfg.confidence_threshold:
            return "unknown", prob
        return self.class_names[idx], prob


def _read_image(path: str) -> np.ndarray:
    abs_path = Path(to_absolute_path(path))
    if not abs_path.exists():
        raise FileNotFoundError(f"Image not found: {abs_path}")
    image = cv2.imread(str(abs_path))
    if image is None:
        raise ValueError(f"Failed to read image: {abs_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_image(cfg: DictConfig):
    logger.info("Starting Image Inference Mode")
    predictor = Predictor(cfg)

    image = _read_image(cfg.input.image_path)
    label, prob = predictor.predict(image)

    print(f"\nResult: {label} (confidence: {prob:.2%})")


def run_webcam(cfg: DictConfig):
    logger.info("Starting Webcam Inference Mode")
    predictor = Predictor(cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        roi_size = 400
        x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
        x2, y2 = x1 + roi_size, y1 + roi_size

        roi = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        label, prob = predictor.predict(roi_rgb)

        color = (0, 255, 0) if prob > cfg.confidence_threshold else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.imshow("ASL Inference", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    if cfg.input.image_path:
        predict_image(cfg)
    else:
        run_webcam(cfg)


if __name__ == "__main__":
    main()
