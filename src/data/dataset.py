"""
EDGE-DR — PyTorch Dataset & Augmentation
=========================================
Provides:
  - DRDataset     : labeled fundus image dataset (CSV-backed)
  - get_transforms: training / validation albumentations pipelines
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

REFERRAL_GRADES = [2, 3, 4]       # → Refer to district eye hospital


# ──────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    """
    Returns albumentations transform pipeline.
    split: 'train' | 'val' | 'test'
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GridDistortion(p=0.1),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(8, 16),
                            hole_width_range=(8, 16),
                            fill=0, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

class DRDataset(Dataset):
    """Diabetic Retinopathy fundus image dataset.

    Args:
        df       : DataFrame with columns ['image_path', 'label']
        transform: albumentations Compose pipeline
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        label = int(row["label"])
        return img, label

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for WeightedRandomSampler / Focal Loss."""
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler."""
        counts = self.df["label"].value_counts().to_dict()
        return [1.0 / counts[int(l)] for l in self.df["label"]]


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloaders(csv_path: str,
                    image_size: int = 224,
                    batch_size: int = 32,
                    val_split: float = 0.15,
                    test_split: float = 0.10,
                    num_workers: int = 4,
                    seed: int = 42) -> dict:
    """
    Build train / val / test DataLoaders from a preprocessed dataset CSV.

    Returns:
        dict with keys 'train', 'val', 'test', 'class_weights'
    """
    df = pd.read_csv(csv_path)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]   # sanity check

    # Stratified splits
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split / (1 - test_split),
        stratify=train_val_df["label"],
        random_state=seed)

    train_ds = DRDataset(train_df, transform=get_transforms("train", image_size))
    val_ds   = DRDataset(val_df,   transform=get_transforms("val",   image_size))
    test_ds  = DRDataset(test_df,  transform=get_transforms("test",  image_size))

    # Balanced sampling for training
    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size,
                            sampler=sampler, num_workers=num_workers,
                            pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True),
        "class_weights": train_ds.class_weights,
        "num_train": len(train_ds),
        "num_val":   len(val_ds),
        "num_test":  len(test_ds),
    }

