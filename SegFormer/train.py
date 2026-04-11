from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter

from .config import SegFormerConfig
from .dataset import build_dataloaders
from .metrics import mean_iou, pixel_accuracy
from .model import SegFormerModule
from .utils import ensure_dir, resolve_device, save_checkpoint, save_config, set_seed

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def parse_args() -> SegFormerConfig:
    parser = argparse.ArgumentParser(description="Train SegFormer on CARLA semantic segmentation data.")
    parser.add_argument("--dataset_dir", type=str, default="vae/my_data_autopilot")
    parser.add_argument("--output_dir", type=str, default="models/segformer")
    parser.add_argument("--log_dir", type=str, default="models/segformer/logs")
    parser.add_argument("--pretrained_model_name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return SegFormerConfig(**vars(args))


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    num_classes=13,
    ignore_index=255,
    stage_name="train",
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_miou = 0.0
    total_acc = 0.0
    start_time = time.time()
    iterator = loader
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(loader, desc=stage_name, leave=False)
        iterator = progress_bar

    for step_idx, batch in enumerate(iterator, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        predictions = logits.argmax(dim=1)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_miou += mean_iou(predictions, labels, num_classes=num_classes, ignore_index=ignore_index)
        total_acc += pixel_accuracy(predictions, labels, ignore_index=ignore_index)

        elapsed = time.time() - start_time
        avg_loss = total_loss / step_idx
        avg_miou = total_miou / step_idx
        avg_acc = total_acc / step_idx
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                mIoU=f"{avg_miou:.4f}",
                acc=f"{avg_acc:.4f}",
                elapsed=f"{elapsed:.1f}s",
            )
        elif step_idx == 1 or step_idx == len(loader):
            print(
                f"  {stage_name} step {step_idx}/{len(loader)} | "
                f"avg_loss={avg_loss:.4f} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()

    denominator = max(1, len(loader))
    epoch_duration = time.time() - start_time
    total_images = len(loader.dataset) if hasattr(loader, "dataset") else denominator
    return {
        "loss": total_loss / denominator,
        "mIoU": total_miou / denominator,
        "pixel_acc": total_acc / denominator,
        "duration_sec": epoch_duration,
        "sec_per_batch": epoch_duration / denominator,
        "sec_per_image": epoch_duration / max(1, total_images),
        "images_per_sec": total_images / max(epoch_duration, 1e-8),
    }


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    log_dir = ensure_dir(config.log_dir)
    save_config(output_dir / "config.json", config.to_dict())
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config", "\n".join(f"{key}: {value}" for key, value in config.to_dict().items()))

    device = resolve_device(config.device)
    train_loader, val_loader = build_dataloaders(config)
    model = SegFormerModule(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    print(f"Using device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    training_start_time = time.time()

    for epoch in range(config.num_epochs):
        print(f"Starting epoch {epoch + 1}/{config.num_epochs}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            stage_name="train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            stage_name="val",
        )
        epoch_duration = train_metrics["duration_sec"] + val_metrics["duration_sec"]
        elapsed_training = time.time() - training_start_time
        avg_epoch_time = elapsed_training / (epoch + 1)
        remaining_epochs = config.num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mIoU={val_metrics['mIoU']:.4f} | "
            f"val_pixel_acc={val_metrics['pixel_acc']:.4f} | "
            f"train_time={train_metrics['duration_sec'] / 60.0:.2f}m | "
            f"val_time={val_metrics['duration_sec'] / 60.0:.2f}m | "
            f"epoch_time={epoch_duration / 60.0:.2f}m | "
            f"eta={eta_seconds / 60.0:.2f}m"
        )

        writer.add_scalar("loss/train", train_metrics["loss"], epoch + 1)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch + 1)
        writer.add_scalar("mIoU/train", train_metrics["mIoU"], epoch + 1)
        writer.add_scalar("mIoU/val", val_metrics["mIoU"], epoch + 1)
        writer.add_scalar("pixel_accuracy/train", train_metrics["pixel_acc"], epoch + 1)
        writer.add_scalar("pixel_accuracy/val", val_metrics["pixel_acc"], epoch + 1)
        writer.add_scalar("time/train_epoch_sec", train_metrics["duration_sec"], epoch + 1)
        writer.add_scalar("time/val_epoch_sec", val_metrics["duration_sec"], epoch + 1)
        writer.add_scalar("time/epoch_total_sec", epoch_duration, epoch + 1)
        writer.add_scalar("time/train_sec_per_image", train_metrics["sec_per_image"], epoch + 1)
        writer.add_scalar("time/val_sec_per_image", val_metrics["sec_per_image"], epoch + 1)
        writer.add_scalar("time/train_images_per_sec", train_metrics["images_per_sec"], epoch + 1)
        writer.add_scalar("time/val_images_per_sec", val_metrics["images_per_sec"], epoch + 1)

        save_checkpoint(output_dir / "last_model.pt", model, optimizer, epoch)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch)

    writer.add_hparams(
        hparam_dict=config.to_dict(),
        metric_dict={
            "hparam/best_val_loss": best_val_loss,
            "hparam/final_train_loss": train_metrics["loss"],
            "hparam/final_val_mIoU": val_metrics["mIoU"],
            "hparam/final_val_pixel_acc": val_metrics["pixel_acc"],
            "hparam/final_train_sec_per_image": train_metrics["sec_per_image"],
            "hparam/final_val_sec_per_image": val_metrics["sec_per_image"],
        },
    )
    writer.close()


if __name__ == "__main__":
    main()
