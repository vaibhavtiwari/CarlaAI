from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from PIL import Image, ImageTk

from models import MlpVAE, ConvVAE
from train_vae import CITYSCAPES_PALETTE, preprocess_rgb_frame, preprocess_seg_frame

parser = argparse.ArgumentParser(description="Visualizes the features learned by the VAE")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_type", type=str, default="cnn")
parser.add_argument("--z_dim", type=int, default=64)
parser.add_argument("--source_shape", type=str, default="160x80x3")
parser.add_argument("--target_shape", type=str, default="160x80x3",
                    help="Remember to set this one to 160x80x1 if model was trained on segmentation maps")
parser.add_argument("--dataset_dir", type=str, default=None,
                    help="Optional dataset root containing rgb/ and segmentation/ for side-by-side comparison")
args = parser.parse_args()

source_shape = np.array([int(x) for x in args.source_shape.split("x")])[[1, 0, 2]]
if args.target_shape == "160x80x3" and "seg_" in args.model_dir:
    args.target_shape = "160x80x1"
target_shape = np.array([int(x) for x in args.target_shape.split("x")])[[1, 0, 2]]

if args.model_type == "cnn": VAEClass = ConvVAE
elif args.model_type == "mlp": VAEClass = MlpVAE    
else: raise Exception("No model type \"{}\"".format(args.model_type))

vae = VAEClass(source_shape=source_shape,
               target_shape=target_shape,
               z_dim=args.z_dim,
               model_dir=args.model_dir,
               training=False)
vae.init_session(init_logging=False)
if not vae.load_latest_checkpoint():
    print("Failed to load latest checkpoint for model \"{}\"".format(args.model_dir))


def render_target_image(image_array):
    image_array = np.asarray(image_array, dtype=np.float32)
    if target_shape[-1] == 1:
        class_ids = np.clip(np.round(image_array.squeeze() * 12), 0, 12).astype(np.int32)
        colored = np.zeros((class_ids.shape[0], class_ids.shape[1], 3), dtype=np.uint8)
        for class_id, color in CITYSCAPES_PALETTE.items():
            colored[class_ids == class_id] = color
        return colored
    return np.clip(image_array * 255.0, 0, 255).astype(np.uint8)


def render_rgb_image(image_array):
    return np.clip(np.asarray(image_array, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def load_rgb_frame(filepath):
    with Image.open(filepath) as image:
        return preprocess_rgb_frame(
            np.asarray(image.resize((source_shape[1], source_shape[0]), resample=Image.BILINEAR))
        )


def load_segmentation_frame(filepath):
    with Image.open(filepath) as image:
        return preprocess_seg_frame(
            np.asarray(image.resize((source_shape[1], source_shape[0]), resample=Image.NEAREST))
        )


def find_ground_truth_path(filepath):
    if args.dataset_dir is not None:
        candidate = os.path.join(args.dataset_dir, "segmentation", os.path.basename(filepath))
        return candidate if os.path.isfile(candidate) else None
    normalized = os.path.normpath(filepath)
    if f"{os.sep}rgb{os.sep}" in normalized:
        candidate = normalized.replace(f"{os.sep}rgb{os.sep}", f"{os.sep}segmentation{os.sep}", 1)
        return candidate if os.path.isfile(candidate) else None
    return None

class UI():
    def __init__(self, z_dim, generate_fn, slider_range=3, image_scale=4):
        # Setup tkinter window
        self.window = Tk()
        self.window.title("VAE Inspector")
        self.window.style = Style()
        self.window.style.theme_use("clam") # ('clam', 'alt', 'default', 'classic')

        self.image_scale = image_scale
        self.generate_fn = generate_fn
        self.current_input = None
        self.current_target = None

        controls = Frame(self.window)
        controls.pack(side=TOP, fill=X, padx=20, pady=(20, 0))

        self.browse = Button(controls, text="Set z by image", command=self.set_z_by_image)
        self.browse.pack(side=LEFT)

        content = Frame(self.window)
        content.pack(side=TOP, fill=BOTH, expand=True, padx=20, pady=20)

        image_panel = Frame(content)
        image_panel.pack(side=LEFT, anchor=N, padx=(0, 30))

        sliders_panel = Frame(content)
        sliders_panel.pack(side=LEFT, anchor=N)

        self.source_image = self._create_image_panel(image_panel, "Input RGB")
        if target_shape[-1] == 1:
            self.target_image = self._create_image_panel(image_panel, "Ground Truth Segmentation")
        else:
            self.target_image = self._create_image_panel(image_panel, "Ground Truth")
        self.prediction_image = self._create_image_panel(image_panel, "Predicted")

        blank_source = np.full(source_shape, 255, dtype=np.uint8)
        blank_target = np.full((target_shape[0], target_shape[1], 3), 255, dtype=np.uint8)
        self._set_image(self.source_image, blank_source, resample=Image.BILINEAR)
        self._set_image(self.target_image, blank_target, resample=Image.NEAREST)
        self._set_image(self.prediction_image, blank_target, resample=Image.NEAREST)

        # Setup sliders for latent vector z
        slider_frames = []
        self.z_vars = [DoubleVar() for _ in range(z_dim)]
        self.update_label_fns = []
        for i in range(z_dim):
            # On slider change event
            def create_slider_event(i, z_i, label):
                def event(_=None, generate=True):
                    label.configure(text="z[{}]={}{:.2f}".format(i, "" if z_i.get() < 0 else " ", z_i.get()))
                    if generate: self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
                return event

            if i % 20 == 0:
                sliders_frame = Frame(sliders_panel)
                slider_frames.append(sliders_frame)

            # Create widgets
            inner_frame = Frame(sliders_frame)
            label = Label(inner_frame, font="TkFixedFont")

            # Create event function
            on_value_changed = create_slider_event(i, self.z_vars[i], label)
            on_value_changed(generate=False) # Call once to set label text
            self.update_label_fns.append(on_value_changed)

            # Create slider
            slider = Scale(inner_frame, value=0.0, variable=self.z_vars[i], orient=HORIZONTAL, length=160,
                        from_=-slider_range, to=slider_range, command=on_value_changed)

            # Pack
            label.pack(side=TOP, anchor=W, padx=6)
            slider.pack(side=TOP, pady=(2, 10))
            inner_frame.pack(side=TOP, anchor=W)
        for f in slider_frames:
            f.pack(side=LEFT, anchor=N, padx=10)

    def _create_image_panel(self, parent, title):
        frame = Frame(parent)
        frame.pack(side=TOP, pady=10)
        Label(frame, text=title).pack(side=TOP, pady=5)
        image_label = Label(frame)
        image_label.pack(side=TOP)
        return image_label

    def set_z_by_image(self):
        filepath = askopenfilename()
        if filepath is not None:
            frame = load_rgb_frame(filepath)
            self.current_input = frame
            self._set_image(self.source_image, render_rgb_image(frame), resample=Image.BILINEAR)

            ground_truth_path = find_ground_truth_path(filepath)
            self.current_target = load_segmentation_frame(ground_truth_path) if ground_truth_path else None
            if self.current_target is not None:
                self._set_image(self.target_image, render_target_image(self.current_target), resample=Image.NEAREST)

            z = vae.sample_latent(np.expand_dims(frame, axis=0))[0]
            for i in range(len(self.z_vars)):
                self.z_vars[i].set(z[i])
                self.update_label_fns[i](generate=False)
            self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))

    def _set_image(self, widget, image_array, resample):
        image_size = (image_array.shape[0] * self.image_scale, image_array.shape[1] * self.image_scale)
        pil_image = Image.fromarray(image_array)
        pil_image = pil_image.resize((image_size[1], image_size[0]), resample=resample)
        tkimage = ImageTk.PhotoImage(image=pil_image)
        widget.image = tkimage
        widget.configure(image=tkimage)

    def update_prediction(self, image_array):
        self._set_image(self.prediction_image, render_target_image(image_array), resample=Image.NEAREST)

    def mainloop(self):
        self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
        self.window.mainloop()

def generate(z):
    generated_image = vae.generate_from_latent(np.expand_dims(z, axis=0))[0].reshape(target_shape)
    ui.update_prediction(generated_image)

ui = UI(vae.sample.shape[1], generate, slider_range=10)
ui.mainloop()
