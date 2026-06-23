from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .config import SegFormerConfig
from .transforms import SegmentationTransform


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def rgb_dir(self) -> Path:
        return self.root / "rgb"

    @property
    def mask_dir(self) -> Path:
        return self.root / "segmentation"


class CarlaSegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str, image_size: tuple[int, int], ignore_index: int = 255):
        self.paths = DatasetPaths(Path(dataset_dir))
        self.transform = SegmentationTransform(image_size=image_size, ignore_index=ignore_index)
        self.filenames = self._discover_filenames()

    def _discover_filenames(self) -> list[str]:
        if not self.paths.rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing rgb directory: {self.paths.rgb_dir}")
        if not self.paths.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing segmentation directory: {self.paths.mask_dir}")

        rgb_filenames = sorted(
            filename for filename in os.listdir(self.paths.rgb_dir) if filename.endswith(".png")
        )
        if not rgb_filenames:
            raise ValueError(f"No PNG images found in {self.paths.rgb_dir}")

        mask_filenames = {filename for filename in os.listdir(self.paths.mask_dir) if filename.endswith(".png")}
        missing = [filename for filename in rgb_filenames if filename not in mask_filenames]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Missing segmentation masks for: {preview}")
        return rgb_filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        filename = self.filenames[index]
        image_path = self.paths.rgb_dir / filename
        mask_path = self.paths.mask_dir / filename

        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            pixel_values, labels = self.transform(image, mask)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "filename": filename,
        }


def build_dataloaders(config: SegFormerConfig) -> tuple[DataLoader, DataLoader]:
    dataset = CarlaSegmentationDataset(
        dataset_dir=config.dataset_dir,
        image_size=config.image_size,
        ignore_index=config.ignore_index,
    )
    val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training samples.")

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
