import glob
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


class ASLDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
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

        self.train_transform = A.Compose(
            [
                A.Resize(self.img_h, self.img_w),
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
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

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset:
            return

        root_dir = Path(self.cfg.data.root_dir)
        all_image_paths = glob.glob(str(root_dir / "*/*.jpg"))

        if not all_image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        all_labels = [class_to_idx[Path(p).parent.name] for p in all_image_paths]

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_image_paths,
            all_labels,
            test_size=(self.cfg.data.val_split + self.cfg.data.test_split),
            stratify=all_labels,
            random_state=self.cfg.seed,
        )

        rel_test_size = self.cfg.data.test_split / (
            self.cfg.data.val_split + self.cfg.data.test_split
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )
