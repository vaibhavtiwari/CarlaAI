from __future__ import annotations

import torch


def pixel_accuracy(predictions: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> float:
    valid = labels != ignore_index
    if valid.sum() == 0:
        return 0.0
    correct = (predictions[valid] == labels[valid]).sum().item()
    total = valid.sum().item()
    return correct / total


def mean_iou(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    ious: list[float] = []
    valid = labels != ignore_index
    predictions = predictions[valid]
    labels = labels[valid]
    if labels.numel() == 0:
        return 0.0

    for class_id in range(num_classes):
        pred_mask = predictions == class_id
        label_mask = labels == class_id
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0
