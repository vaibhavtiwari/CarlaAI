from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

from .visualize import encode_segmentation_mask


def resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, resample=Image.BILINEAR)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.resize(size, resample=Image.NEAREST)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.int64)
    if array.ndim == 3:
        array = array[:, :, 0]
    return torch.from_numpy(array)


def _fit_with_aspect_ratio(image: Image.Image, size: tuple[int, int], resample) -> tuple[Image.Image, tuple[int, int]]:
    src_width, src_height = image.size
    target_width, target_height = size
    scale = min(target_width / src_width, target_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    resized = image.resize((resized_width, resized_height), resample=resample)
    return resized, (resized_width, resized_height)


def _pad_to_size(
    image: Image.Image,
    size: tuple[int, int],
    fill,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    target_width, target_height = size
    pad_width = target_width - image.size[0]
    pad_height = target_height - image.size[1]
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    padded = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)
    return padded, (left, top, right, bottom)


def prepare_image(
    image: Image.Image,
    size: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, tuple[int, int] | tuple[int, int, int, int]]]:
    resized, resized_size = _fit_with_aspect_ratio(image.convert("RGB"), size, Image.BILINEAR)
    padded, padding = _pad_to_size(resized, size, fill=(0, 0, 0))
    return image_to_tensor(padded), {
        "resized_size": resized_size,
        "padding": padding,
        "target_size": size,
    }


def prepare_mask(mask: Image.Image, size: tuple[int, int], ignore_index: int) -> torch.Tensor:
    encoded = encode_segmentation_mask(np.asarray(mask), unknown_value=ignore_index).astype(np.uint8)
    encoded_image = Image.fromarray(encoded, mode="L")
    resized, _ = _fit_with_aspect_ratio(encoded_image, size, Image.NEAREST)
    padded, _ = _pad_to_size(resized, size, fill=ignore_index)
    return mask_to_tensor(padded)


def crop_prediction(prediction: np.ndarray, metadata: dict[str, tuple[int, int] | tuple[int, int, int, int]]) -> np.ndarray:
    left, top, right, bottom = metadata["padding"]
    height, width = prediction.shape[:2]
    cropped = prediction[top: height - bottom if bottom > 0 else height, left: width - right if right > 0 else width]
    resized_width, resized_height = metadata["resized_size"]
    if cropped.shape[1] != resized_width or cropped.shape[0] != resized_height:
        cropped = np.asarray(
            Image.fromarray(cropped.astype(np.uint8)).resize((resized_width, resized_height), resample=Image.NEAREST)
        )
    return cropped


@dataclass
class SegmentationTransform:
    image_size: tuple[int, int]
    ignore_index: int = 255

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor, _ = prepare_image(image, self.image_size)
        mask_tensor = prepare_mask(mask, self.image_size, self.ignore_index)
        return image_tensor, mask_tensor
