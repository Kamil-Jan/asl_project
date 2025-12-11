import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataset import ASLDataset


class ASLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        img_h = self.cfg.data.img_size[0]
        img_w = self.cfg.data.img_size[1]

        self.train_transform = A.Compose([
            A.Resize(img_h, img_w),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.06, 0.06),
                rotate=(-20, 20),
                p=0.5
            ),

            A.HorizontalFlip(p=0.5),

            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),

            A.GaussianBlur(p=0.3),

            A.GaussNoise(p=0.3),

            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                p=0.3
            ),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(img_h, img_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset:
            return

        root_dir = Path(self.cfg.data.root_dir)
        all_image_paths = glob.glob(str(root_dir / "*/*.jpg"))

        if not all_image_paths:
             raise FileNotFoundError(f"Картинки не найдены в {root_dir}. Проверь путь в конфиге!")

        classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        all_labels = []
        for path in all_image_paths:
            parent_name = Path(path).parent.name
            all_labels.append(class_to_idx[parent_name])

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=(self.cfg.data.val_split + self.cfg.data.test_split),
            stratify=all_labels,
            random_state=self.cfg.seed
        )

        relative_test_size = self.cfg.data.test_split / (self.cfg.data.val_split + self.cfg.data.test_split)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=relative_test_size,
            stratify=temp_labels,
            random_state=self.cfg.seed
        )

        self.train_dataset = ASLDataset(train_paths, train_labels, transform=self.train_transform)
        self.val_dataset = ASLDataset(val_paths, val_labels, transform=self.val_transform)
        self.test_dataset = ASLDataset(test_paths, test_labels, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=True, num_workers=self.cfg.data.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=False, num_workers=self.cfg.data.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=False, num_workers=self.cfg.data.num_workers, persistent_workers=True)
