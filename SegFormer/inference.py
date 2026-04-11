from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from .config import SegFormerConfig
from .model import SegFormerModule
from .transforms import crop_prediction, prepare_image
from .utils import resolve_device
from .visualize import decode_segmentation_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegFormer inference on a single image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="segformer_prediction.png")
    parser.add_argument("--pretrained_model_name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SegFormerConfig(
        pretrained_model_name=args.pretrained_model_name,
        image_width=args.image_width,
        image_height=args.image_height,
        num_classes=args.num_classes,
        device=args.device,
    )

    device = resolve_device(config.device)
    model = SegFormerModule(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(args.image) as image:
        original_rgb = image.convert("RGB")
        pixel_tensor, metadata = prepare_image(original_rgb, config.image_size)
        pixel_values = pixel_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = interpolate(
            outputs.logits,
            size=(config.image_height, config.image_width),
            mode="bilinear",
            align_corners=False,
        )
        predictions = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    predictions = crop_prediction(predictions, metadata)
    colored = decode_segmentation_mask(predictions)
    colored = np.asarray(
        Image.fromarray(colored).resize(original_rgb.size, resample=Image.NEAREST)
    )
    Image.fromarray(np.asarray(colored)).save(Path(args.output))
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
