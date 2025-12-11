import albumentations as A
import cv2
import hydra
import numpy as np
import onnxruntime as ort
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from asl_project.model import ASLModule


class Predictor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = cfg.class_names
        self.img_h, self.img_w = cfg.img_size

        self.transform = A.Compose(
            [
                A.Resize(self.img_h, self.img_w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        if cfg.model_type == "onnx":
            print(f"Loading ONNX session: {cfg.onnx_path}")
            self.ort_session = ort.InferenceSession(
                cfg.onnx_path, providers=["CPUExecutionProvider"]
            )
        else:
            print(f"Loading Torch checkpoint: {cfg.checkpoint_path}")
            self.model = ASLModule.load_from_checkpoint(cfg.checkpoint_path).to(
                self.device
            )
            self.model.eval()
            self.model.freeze()

    def preprocess(self, image: np.ndarray):
        augmented = self.transform(image=image)
        return augmented["image"].unsqueeze(0)

    def predict(self, image: np.ndarray):
        tensor = self.preprocess(image)

        if self.cfg.model_type == "onnx":
            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: tensor.numpy()}
            logits = self.ort_session.run(None, ort_inputs)[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            idx = np.argmax(probs)
            prob = probs[0][idx]
        else:
            with torch.no_grad():
                tensor = tensor.to(self.device)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                prob, idx = torch.max(probs, dim=1)
                idx = idx.item()
                prob = prob.item()

        return self.class_names[idx], prob


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run_webcam(cfg: DictConfig):
    predictor = Predictor(cfg)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        label, prob = predictor.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {prob: .2f}",
            (100, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("ASL Inference", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
