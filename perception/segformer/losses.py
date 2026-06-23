from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
