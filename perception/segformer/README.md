# SegFormer

Modular training package for CARLA semantic segmentation.

## Files

- `config.py`: central hyperparameters and paths
- `dataset.py`: dataset discovery and dataloader creation
- `transforms.py`: image and mask preprocessing
- `model.py`: SegFormer wrapper around Hugging Face transformers
- `losses.py`: loss helpers
- `metrics.py`: pixel accuracy and mIoU
- `utils.py`: checkpointing, seed setup, and device resolution
- `train.py`: training entry point
- `evaluation.py`: validation entry point for saved checkpoints
- `inference.py`: prediction on single images
- `inspect_segformer.py`: interactive viewer with file picker
- `visualize.py`: segmentation mask decoding

## Expected dataset layout

```text
dataset_root/
├── rgb/
└── segmentation/
```

Filenames should match across both folders.

## Checkpoints

Training writes only:

- `best_model.pt`: updated when validation loss improves
- `last_model.pt`: overwritten every epoch for resume/debug use

## TensorBoard

Training also writes TensorBoard logs to `models/segformer/logs` by default.

Logged information includes:

- hyperparameters for the run
- training and validation loss curves
- training and validation mIoU curves
- training and validation pixel accuracy curves

## Preprocessing

The default training size is `512 x 512`, but images are not stretched directly.

- input RGB frames are resized with aspect ratio preserved
- the resized image is padded to `512 x 512`
- segmentation masks use the same resize-plus-padding path
- padded mask regions are filled with the ignore label so they do not affect loss

Example:

```bash
tensorboard --logdir models/segformer/logs
```

## Example commands

```bash
python -m perception.segformer.train --dataset_dir perception/vae/my_data_autopilot
python -m perception.segformer.evaluation --checkpoint models/segformer/best_model.pt
python -m perception.segformer.inference --checkpoint models/segformer/best_model.pt --image images/rgb/example.png
python -m perception.segformer.inspect_segformer --checkpoint models/segformer/best_model.pt --dataset_dir perception/vae/my_data_autopilot
```
