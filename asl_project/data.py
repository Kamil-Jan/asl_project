from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from asl_project.utils import ensure_data_ready, infer_class_names


class ASLDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class ASLDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.img_h, self.img_w = cfg.data.img_size
        self.train_dir = Path(cfg.data.train_dir)
        self.class_names = list(cfg.data.class_names) if cfg.data.class_names else []

        self.train_transform = A.Compose(
            [
                A.Resize(self.img_h, self.img_w),
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Resize(self.img_h, self.img_w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset:
            return

        dvc_target = (
            self.cfg.paths.dvc_target
            if hasattr(self.cfg, "paths") and hasattr(self.cfg.paths, "dvc_target")
            else None
        )
        ensure_data_ready(self.cfg.data, dvc_target=dvc_target)

        extensions = {"*.jpg", "*.png", "*.jpeg"}
        all_image_paths = []
        for ext in extensions:
            all_image_paths.extend(self.train_dir.glob(f"**/{ext}"))

        if not all_image_paths:
            raise FileNotFoundError(f"No images found in {self.train_dir}")

        if not self.class_names:
            self.class_names = infer_class_names(self.train_dir)

        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        valid_paths = []
        all_labels = []

        for p in all_image_paths:
            parent_name = p.parent.name
            if parent_name in class_to_idx:
                valid_paths.append(p)
                all_labels.append(class_to_idx[parent_name])

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            valid_paths,
            all_labels,
            test_size=(self.cfg.data.val_split + self.cfg.data.test_split),
            stratify=all_labels,
            random_state=self.cfg.seed,
        )

        temp_split_size = self.cfg.data.val_split + self.cfg.data.test_split
        rel_test_size = (
            self.cfg.data.test_split / temp_split_size if temp_split_size > 0 else 0
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=rel_test_size,
            stratify=temp_labels,
            random_state=self.cfg.seed,
        )

        self.train_dataset = ASLDataset(train_paths, train_labels, self.train_transform)
        self.val_dataset = ASLDataset(val_paths, val_labels, self.val_transform)
        self.test_dataset = ASLDataset(test_paths, test_labels, self.val_transform)

    def train_dataloader(self):
        is_mps = torch.backends.mps.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
            pin_memory=False if is_mps else True,
        )

    def val_dataloader(self):
        is_mps = torch.backends.mps.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
            pin_memory=False if is_mps else True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )
