from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from perception.segformer.config import SegFormerConfig
from perception.segformer.model import SegFormerModule
from perception.segformer.transforms import prepare_image
from perception.segformer.utils import resolve_device


class SegFormerEncoder:
    def __init__(self, model: SegFormerModule, config: SegFormerConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.feature_dim = config.num_classes

    def encode(self, frame: np.ndarray) -> np.ndarray:
        with Image.fromarray(frame.astype(np.uint8), mode="RGB") as image:
            pixel_tensor, _ = prepare_image(image, self.config.image_size)

        pixel_values = pixel_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probabilities = torch.softmax(outputs.logits, dim=1)
            pooled = F.adaptive_avg_pool2d(probabilities, output_size=1).flatten(start_dim=1)
        return pooled.squeeze(0).cpu().numpy()


def load_segformer(
    checkpoint_path: str,
    pretrained_model_name: str,
    image_size: tuple[int, int] = (512, 512),
    num_classes: int = 13,
    device: str = "cuda",
) -> SegFormerEncoder:
    config = SegFormerConfig(
        pretrained_model_name=pretrained_model_name,
        image_width=image_size[0],
        image_height=image_size[1],
        num_classes=num_classes,
        device=device,
    )
    resolved_device = resolve_device(config.device)
    model = SegFormerModule(config).to(resolved_device)

    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return SegFormerEncoder(model=model, config=config, device=resolved_device)


def create_encode_state_fn(segformer: SegFormerEncoder, measurements_to_include):
    measure_flags = [
        "steer" in measurements_to_include,
        "throttle" in measurements_to_include,
        "speed" in measurements_to_include,
        "orientation" in measurements_to_include,
    ]

    def encode_state(env):
        encoded_state = segformer.encode(env.observation)

        measurements = []
        if measure_flags[0]:
            measurements.append(env.vehicle.control.steer)
        if measure_flags[1]:
            measurements.append(env.vehicle.control.throttle)
        if measure_flags[2]:
            measurements.append(env.vehicle.get_speed())
        if measure_flags[3]:
            forward = env.vehicle.get_forward_vector()
            measurements.extend([forward.x, forward.y, forward.z])

        return np.append(encoded_state, measurements)

    return encode_state
