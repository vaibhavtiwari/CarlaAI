from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from .config import SegFormerConfig
from .dataset import build_dataloaders
from .model import SegFormerModule
from .train import run_epoch
from .utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SegFormer checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="perception/vae/my_data_autopilot")
    parser.add_argument("--pretrained_model_name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SegFormerConfig(
        dataset_dir=args.dataset_dir,
        pretrained_model_name=args.pretrained_model_name,
        image_width=args.image_width,
        image_height=args.image_height,
        num_classes=args.num_classes,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    device = resolve_device(config.device)
    _, val_loader = build_dataloaders(config)
    model = SegFormerModule(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metrics = run_epoch(
        model,
        val_loader,
        device,
        optimizer=None,
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
    )
    print(
        f"Validation | loss={metrics['loss']:.4f} | "
        f"mIoU={metrics['mIoU']:.4f} | pixel_acc={metrics['pixel_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
