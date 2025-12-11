import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from src.baseline_model import BaselineCNN
from typing import Tuple


class ASLPredictor:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        self.cfg = cfg

        self.model = BaselineCNN.load_from_checkpoint(
            cfg.inference.checkpoint_path,
            map_location=self.device
        )
        self.model.eval()
        self.model.freeze()

        img_h = self.cfg.data.img_size[0]
        img_w = self.cfg.data.img_size[1]

        self.transform = A.Compose([
            A.Resize(img_h, img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        augmented = self.transform(image=image)
        tensor_img = augmented["image"].unsqueeze(0).to(self.device)

        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

        return self.classes[top_idx.item()], top_prob.item()
