import argparse
import os
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from models import ConvVAE, MlpVAE, bce_loss, bce_loss_v2, mse_loss

CITYSCAPES_PALETTE = {
    0: (0, 0, 0),        # None
    1: (70, 70, 70),     # Buildings
    2: (190, 153, 153),  # Fences
    3: (72, 0, 90),      # Other
    4: (220, 20, 60),    # Pedestrians
    5: (153, 153, 153),  # Poles
    6: (157, 234, 50),   # RoadLines
    7: (128, 64, 128),   # Roads
    8: (244, 35, 232),   # Sidewalks
    9: (107, 142, 35),   # Vegetation
    10: (0, 0, 255),     # Vehicles
    11: (102, 102, 156), # Walls
    12: (220, 220, 0),   # TrafficSigns
}

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value}")

def parse_image_size(value):
    width, height = (int(x) for x in value.split("x"))
    return width, height

def preprocess_rgb_frame(frame):
    frame = frame[:, :, :3]                  # RGBA -> RGB
    frame = frame.astype(np.float32) / 255.0 # [0, 255] -> [0, 1]
    return frame

def preprocess_seg_frame_road_only(frame):
    frame = frame[:, :, :1]                 # RGBA -> R
    frame = frame == 7                      # Create a binary mask for the road class
    frame = frame.astype(np.float32)        # To float
    return frame

def preprocess_seg_frame(frame):
    if frame[:, :, :3].max() <= 12:
        frame = frame[:, :, :1]                 # RGBA -> R
        return frame.astype(np.float32) / 12.0  # [0, 12=num_classes] -> [0, 1]

    rgb_frame = frame[:, :, :3]
    class_ids = np.zeros(rgb_frame.shape[:2], dtype=np.float32)
    for class_id, color in CITYSCAPES_PALETTE.items():
        class_ids[np.all(rgb_frame == color, axis=-1)] = class_id
    return np.expand_dims(class_ids / 12.0, axis=-1)

class CarlaVaeDataset(Dataset):
    def __init__(self, dataset_dir, use_segmentation_as_target=False, image_size=(160, 80)):
        self.dataset_dir = dataset_dir
        self.use_segmentation_as_target = use_segmentation_as_target
        self.image_size = image_size
        self.rgb_dir = os.path.join(dataset_dir, "rgb")
        self.segmentation_dir = os.path.join(dataset_dir, "segmentation")

        self.rgb_filenames = self._list_pngs(self.rgb_dir)
        if not self.rgb_filenames:
            raise ValueError(f"No PNG images found in {self.rgb_dir}")

        if self.use_segmentation_as_target:
            seg_filenames = set(self._list_pngs(self.segmentation_dir))
            missing = sorted(set(self.rgb_filenames) - seg_filenames)
            if missing:
                raise ValueError(f"Missing segmentation files for RGB frames: {missing[:5]}")

        self.source_shape = self._load_source(0).shape
        self.target_shape = self._load_target(0).shape if self.use_segmentation_as_target else self.source_shape

    def _list_pngs(self, dir_path):
        return sorted(filename for filename in os.listdir(dir_path)
                      if os.path.splitext(filename)[1] == ".png")

    def _load_image(self, dir_path, filename, preprocess_fn, resample=Image.BILINEAR):
        filepath = os.path.join(dir_path, filename)
        with Image.open(filepath) as image:
            if self.image_size is not None:
                image = image.resize(self.image_size, resample=resample)
            return preprocess_fn(np.asarray(image))

    def _load_source(self, index):
        return self._load_image(self.rgb_dir, self.rgb_filenames[index], preprocess_rgb_frame)

    def _load_target(self, index):
        if not self.use_segmentation_as_target:
            return self._load_source(index)
        return self._load_image(self.segmentation_dir,
                                self.rgb_filenames[index],
                                preprocess_seg_frame,
                                resample=Image.NEAREST)

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, index):
        source = self._load_source(index)
        if not self.use_segmentation_as_target:
            return source, source
        return source, self._load_target(index)

def train_val_split(dataset, val_portion=0.1):
    val_size = int(len(dataset) * val_portion)
    if val_size <= 0:
        raise ValueError("Validation split is empty. Add more images or reduce val_portion.")
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Add more images or reduce val_portion.")
    generator = torch.Generator().manual_seed(0)
    return random_split(dataset, [train_size, val_size], generator=generator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a VAE with RGB images as source and RGB or segmentation images as target")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--use_segmentation_as_target", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=parse_image_size, default=(160, 80))
    parser.add_argument("--kl_tolerance", type=float, default=0.0)
    parser.add_argument("-restart", action="store_true")
    args = parser.parse_args()

    # Load image pairs from dataset/rgb and dataset/segmentation folders
    # (for every rgb/xxx.png we expect a corresponding segmentation/xxx.png)
    dataset = CarlaVaeDataset(args.dataset, args.use_segmentation_as_target, args.image_size)
    train_dataset, val_dataset = train_val_split(dataset, val_portion=0.1)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    # Get source and target image sizes
    # (may be different e.g. RGB and grayscale)
    source_shape = dataset.source_shape
    target_shape = dataset.target_shape

    # Set model name from params
    if args.model_name is None:
        dataset_name = os.path.basename(os.path.normpath(args.dataset))
        args.model_name = "{}_{}_{}_zdim{}_beta{}_kl_tolerance{}_{}".format(
            "seg" if args.use_segmentation_as_target else "rgb",
            args.loss_type, args.model_type, args.z_dim, args.beta, args.kl_tolerance,
            os.path.splitext(dataset_name)[0])

    print("train_source_images.shape", (len(train_dataset), *source_shape))
    print("val_source_images.shape", (len(val_dataset), *source_shape))
    print("train_target_images.shape", (len(train_dataset), *target_shape))
    print("val_target_images.shape", (len(val_dataset), *target_shape))
    print("")
    print("Training parameters:")
    for k, v, in vars(args).items(): print(f"  {k}: {v}")
    print("")

    if args.loss_type == "bce": loss_fn = bce_loss
    elif args.loss_type == "bce_v2": loss_fn = bce_loss_v2
    elif args.loss_type == "mse": loss_fn = mse_loss
    else: raise Exception("No loss function \"{}\"".format(args.loss_type))

    if args.model_type == "cnn": VAEClass = ConvVAE
    elif args.model_type == "mlp": VAEClass = MlpVAE    
    else: raise Exception("No model type \"{}\"".format(args.model_type))

    # Create VAE model
    vae = VAEClass(source_shape=source_shape,
                   target_shape=target_shape,
                   z_dim=args.z_dim,
                   beta=args.beta,
                   learning_rate=args.learning_rate,
                   lr_decay=args.lr_decay,
                   kl_tolerance=args.kl_tolerance,
                   loss_fn=loss_fn,
                   model_dir=os.path.join("models", args.model_name))
    print(f"Using device: {vae.device}")

    # Prompt to load existing model if any
    if not args.restart:
        if os.path.isdir(vae.log_dir) and len(os.listdir(vae.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(args.model_name))
            if answer.upper() == "C":
                pass
            elif answer.upper() == "R":
                args.restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(args.model_name))
    
    if args.restart:
        if os.path.isdir(vae.model_dir):
            shutil.rmtree(vae.model_dir)
        for d in vae.dirs:
            os.makedirs(d)
    vae.init_session()
    if not args.restart:
        if not vae.load_latest_checkpoint():
            print("No PyTorch VAE checkpoint found; starting from scratch")

    # Training loop
    min_val_loss = float("inf")
    counter = 0
    print("Training")
    while True:
        epoch = vae.get_step_idx()
        print(f"Epoch {epoch + 1}", flush=True)

        # Train one epoch
        vae.train_one_epoch(train_loader, progress=True)

        # Calculate evaluation metrics
        val_loss, val_kl_loss = vae.evaluate(val_loader, progress=True)
        print(f"  val loss={val_loss:.4f}, val kl={val_kl_loss:.4f}", flush=True)
        
        # Early stopping
        if val_loss < min_val_loss:
            counter = 0
            min_val_loss = val_loss
            vae.save() # Save if better
        else:
            counter += 1
            if counter >= 10:
                print("No improvement in last 10 epochs, stopping")
                break
