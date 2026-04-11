from __future__ import annotations

import numpy as np

from .config import CARLA_COLOR_TO_CLASS_ID, CLASS_PALETTE


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    if mask.ndim == 3:
        mask = mask.squeeze()
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items():
        canvas[mask == class_id] = color
    return canvas


def overlay_segmentation(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    colored_mask = decode_segmentation_mask(mask).astype(np.float32)
    blended = (1.0 - alpha) * image + alpha * colored_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def encode_segmentation_mask(mask: np.ndarray, unknown_value: int) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask.astype(np.int64)

    if mask.ndim != 3 or mask.shape[2] < 3:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    rgb_mask = mask[:, :, :3]
    encoded = np.full(rgb_mask.shape[:2], unknown_value, dtype=np.int64)
    for color, class_id in CARLA_COLOR_TO_CLASS_ID.items():
        encoded[np.all(rgb_mask == color, axis=-1)] = class_id
    return encoded
