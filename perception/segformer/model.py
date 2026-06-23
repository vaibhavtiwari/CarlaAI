from __future__ import annotations

import torch
from torch import nn

from .config import SegFormerConfig

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:  # pragma: no cover - handled at runtime by user environment
    SegformerForSemanticSegmentation = None


class SegFormerModule(nn.Module):
    def __init__(self, config: SegFormerConfig):
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise ImportError(
                "transformers is required for SegFormerModule. Install it with `pip install transformers`."
            )

        self.config = config
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            config.pretrained_model_name,
            num_labels=config.num_classes,
            id2label=config.id2label,
            label2id=config.label2id,
            semantic_loss_ignore_index=config.ignore_index,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        return self.model(pixel_values=pixel_values, labels=labels)
