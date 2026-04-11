from dataclasses import asdict, dataclass
from pathlib import Path

CLASS_NAMES = [
    "None",
    "Buildings",
    "Fences",
    "Other",
    "Pedestrians",
    "Poles",
    "RoadLines",
    "Roads",
    "Sidewalks",
    "Vegetation",
    "Vehicles",
    "Walls",
    "TrafficSigns",
]

CLASS_PALETTE = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (190, 153, 153),
    3: (72, 0, 90),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 255),
    11: (102, 102, 156),
    12: (220, 220, 0),
}

CARLA_COLOR_TO_CLASS_ID = {
    (0, 0, 0): 0,
    (70, 70, 70): 1,
    (190, 153, 153): 2,
    (72, 0, 90): 3,
    (220, 20, 60): 4,
    (153, 153, 153): 5,
    (157, 234, 50): 6,
    (128, 64, 128): 7,
    (244, 35, 232): 8,
    (107, 142, 35): 9,
    (152, 251, 152): 9,
    (0, 0, 142): 10,
    (0, 0, 230): 10,
    (119, 11, 32): 10,
    (102, 102, 156): 11,
    (220, 220, 0): 12,
    (250, 170, 30): 12,
    (70, 130, 180): 3,
    (81, 0, 81): 3,
    (150, 100, 100): 3,
    (230, 150, 140): 3,
    (180, 165, 180): 3,
    (110, 190, 160): 3,
    (170, 120, 50): 3,
    (55, 90, 80): 3,
}


@dataclass
class SegFormerConfig:
    dataset_dir: str = "vae/my_data_autopilot"
    output_dir: str = "models/segformer"
    log_dir: str = "models/segformer/logs"
    pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    image_width: int = 512
    image_height: int = 512
    num_classes: int = 13
    ignore_index: int = 255
    batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 6e-5
    weight_decay: float = 1e-2
    num_epochs: int = 20
    val_split: float = 0.1
    num_workers: int = 0
    seed: int = 0
    device: str = "cuda"

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def id2label(self) -> dict[int, str]:
        return {index: name for index, name in enumerate(CLASS_NAMES[: self.num_classes])}

    @property
    def label2id(self) -> dict[str, int]:
        return {name: index for index, name in self.id2label.items()}

    def to_dict(self) -> dict:
        return asdict(self)
