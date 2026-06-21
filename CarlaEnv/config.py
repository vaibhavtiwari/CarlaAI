import argparse
import json
from dataclasses import asdict, dataclass


@dataclass
class SimulatorConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    synchronous: bool = True
    fps: int = 30
    start_carla: bool = False


@dataclass
class DisplayConfig:
    viewer_res: tuple[int, int] = (1280, 720)
    obs_res: tuple[int, int] = (160, 80)
    show_waypoints: bool = True


@dataclass
class CollectorConfig:
    num_images: int = 10000
    output_dir: str = "images"
    target_speed: float = 20.0
    frame_skip: int = 2
    min_save_distance: float = 1.0
    min_route_distance: float = 100.0


def parse_resolution(value):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    raise ValueError(f"Unsupported resolution value: {value!r}")


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args_with_config(parser: argparse.ArgumentParser):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.config:
        config_data = load_json_config(pre_args.config)
        parser.set_defaults(**config_data)

    args = parser.parse_args(remaining)
    setattr(args, "config", pre_args.config)
    return args


def namespace_to_env_config(args):
    simulator = SimulatorConfig(
        host=getattr(args, "host", "127.0.0.1"),
        port=int(getattr(args, "port", 2000)),
        synchronous=bool(getattr(args, "synchronous", True)),
        fps=int(getattr(args, "fps", 30)),
        start_carla=bool(getattr(args, "start_carla", False)),
    )
    display = DisplayConfig(
        viewer_res=parse_resolution(getattr(args, "viewer_res", "1280x720")),
        obs_res=parse_resolution(getattr(args, "obs_res", "160x80")),
        show_waypoints=bool(getattr(args, "show_waypoints", True)),
    )
    return simulator, display


def dataclass_dict(instance):
    return asdict(instance)
