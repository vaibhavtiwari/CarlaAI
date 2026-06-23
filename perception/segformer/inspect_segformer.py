from __future__ import annotations

import argparse
import os
from tkinter import BOTH, LEFT, N, TOP, X, Tk
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Button, Frame, Label, Style

import numpy as np
import torch
from PIL import Image, ImageTk
from torch.nn.functional import interpolate

from .config import SegFormerConfig
from .model import SegFormerModule
from .transforms import crop_prediction, prepare_image, resize_mask, resize_rgb
from .utils import resolve_device
from .visualize import decode_segmentation_mask, encode_segmentation_mask, overlay_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive SegFormer inspector.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--pretrained_model_name", type=str, default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_scale", type=int, default=1)
    return parser.parse_args()


def load_model(config: SegFormerConfig, checkpoint_path: str):
    device = resolve_device(config.device)
    model = SegFormerModule(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict_mask(model, device, image: Image.Image, config: SegFormerConfig) -> np.ndarray:
    pixel_tensor, metadata = prepare_image(image, config.image_size)
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
    return crop_prediction(predictions, metadata)


def find_ground_truth_path(filepath: str, dataset_dir: str | None) -> str | None:
    if dataset_dir is not None:
        candidate = os.path.join(dataset_dir, "segmentation", os.path.basename(filepath))
        return candidate if os.path.isfile(candidate) else None
    normalized = os.path.normpath(filepath)
    if f"{os.sep}rgb{os.sep}" in normalized:
        candidate = normalized.replace(f"{os.sep}rgb{os.sep}", f"{os.sep}segmentation{os.sep}", 1)
        return candidate if os.path.isfile(candidate) else None
    return None


class SegFormerInspector:
    def __init__(self, model, device, config: SegFormerConfig, dataset_dir: str | None, image_scale: int):
        self.model = model
        self.device = device
        self.config = config
        self.dataset_dir = dataset_dir
        self.image_scale = image_scale

        self.window = Tk()
        self.window.title("SegFormer Inspector")
        self.window.style = Style()
        self.window.style.theme_use("clam")

        controls = Frame(self.window)
        controls.pack(side=TOP, fill=X, padx=20, pady=(20, 0))

        Button(controls, text="Choose RGB Image", command=self.choose_image).pack(side=LEFT)

        content = Frame(self.window)
        content.pack(side=TOP, fill=BOTH, expand=True, padx=20, pady=20)

        top_row = Frame(content)
        top_row.pack(side=TOP, anchor=N, pady=(0, 12))
        bottom_row = Frame(content)
        bottom_row.pack(side=TOP, anchor=N)

        self.input_panel = self._create_image_panel(top_row, "Input RGB")
        self.target_panel = self._create_image_panel(top_row, "Ground Truth Segmentation")
        self.prediction_panel = self._create_image_panel(bottom_row, "Prediction")
        self.overlay_panel = self._create_image_panel(bottom_row, "Overlay")

        blank_rgb = np.full((config.image_height, config.image_width, 3), 255, dtype=np.uint8)
        blank_mask = np.full((config.image_height, config.image_width, 3), 255, dtype=np.uint8)
        self._set_image(self.input_panel, blank_rgb, resample=Image.BILINEAR)
        self._set_image(self.target_panel, blank_mask, resample=Image.NEAREST)
        self._set_image(self.prediction_panel, blank_mask, resample=Image.NEAREST)
        self._set_image(self.overlay_panel, blank_rgb, resample=Image.BILINEAR)

    def _create_image_panel(self, parent, title: str):
        frame = Frame(parent)
        frame.pack(side=LEFT, anchor=N, padx=12)
        Label(frame, text=title).pack(side=TOP, pady=5)
        image_label = Label(frame)
        image_label.pack(side=TOP)
        return image_label

    def _set_image(self, widget, image_array: np.ndarray, resample):
        image_size = (image_array.shape[0] * self.image_scale, image_array.shape[1] * self.image_scale)
        pil_image = Image.fromarray(image_array)
        pil_image = pil_image.resize((image_size[1], image_size[0]), resample=resample)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        widget.image = tk_image
        widget.configure(image=tk_image)

    def choose_image(self):
        filepath = askopenfilename()
        if not filepath:
            return

        with Image.open(filepath) as image:
            resized_rgb = resize_rgb(image, self.config.image_size)
            rgb_array = np.asarray(resized_rgb.convert("RGB"), dtype=np.uint8)
            prediction_mask = predict_mask(self.model, self.device, image, self.config)
            prediction_vis = np.asarray(
                Image.fromarray(decode_segmentation_mask(prediction_mask)).resize(
                    self.config.image_size,
                    resample=Image.NEAREST,
                )
            )

        self._set_image(self.input_panel, rgb_array, resample=Image.BILINEAR)
        self._set_image(self.prediction_panel, prediction_vis, resample=Image.NEAREST)
        self._set_image(
            self.overlay_panel,
            overlay_segmentation(np.asarray(resized_rgb.convert("RGB"), dtype=np.uint8), np.asarray(Image.fromarray(prediction_mask.astype(np.uint8)).resize(self.config.image_size, resample=Image.NEAREST))),
            resample=Image.BILINEAR,
        )

        ground_truth_path = find_ground_truth_path(filepath, self.dataset_dir)
        if ground_truth_path is None:
            blank = np.full((self.config.image_height, self.config.image_width, 3), 255, dtype=np.uint8)
            self._set_image(self.target_panel, blank, resample=Image.NEAREST)
            return

        with Image.open(ground_truth_path) as mask:
            resized_mask = resize_mask(mask, self.config.image_size)
            mask_array = encode_segmentation_mask(np.asarray(resized_mask), unknown_value=self.config.ignore_index)
        self._set_image(self.target_panel, decode_segmentation_mask(mask_array), resample=Image.NEAREST)

    def mainloop(self):
        self.window.mainloop()


def main() -> None:
    args = parse_args()
    config = SegFormerConfig(
        pretrained_model_name=args.pretrained_model_name,
        image_width=args.image_width,
        image_height=args.image_height,
        num_classes=args.num_classes,
        device=args.device,
    )
    model, device = load_model(config, args.checkpoint)
    ui = SegFormerInspector(
        model=model,
        device=device,
        config=config,
        dataset_dir=args.dataset_dir,
        image_scale=args.image_scale,
    )
    ui.mainloop()


if __name__ == "__main__":
    main()
