# CarlaAI

CarlaAI is a CARLA-based autonomous driving research repository that combines:

- PPO training for end-to-end driving agents
- VAE-based state representation learning
- SegFormer-based semantic segmentation with transfer learning
- interactive inspection tools for learned reconstructions and segmentation outputs

The repo is centered around CARLA driving experiments, with support for data collection, perception model training, policy training, evaluation, and visualization.

## What This Repo Provides

- Gym-like CARLA environments for lap-following and route-following tasks
- PPO training and evaluation scripts for autonomous driving agents
- VAE training on RGB or semantic segmentation targets
- SegFormer fine-tuning on CARLA semantic segmentation data
- TensorBoard logging for perception model experiments
- Interactive inspectors for VAE reconstructions and SegFormer predictions
- Utilities for collecting paired RGB and segmentation data from CARLA

## Main Components

### PPO and CARLA Environments

- [train.py](train.py): train a PPO driving agent
- [run_eval.py](run_eval.py): run a trained agent in evaluation mode
- [ppo.py](ppo.py): PPO model definition
- [reward_functions.py](reward_functions.py): reward shaping logic
- [CarlaEnv/](CarlaEnv): CARLA environments, data collection, planners, wrappers, and HUD code

### VAE Workflow

- [vae/train_vae.py](vae/train_vae.py): train a VAE on RGB frames or segmentation targets
- [vae/inspect_vae.py](vae/inspect_vae.py): interactive viewer for VAE reconstructions
- [vae/models.py](vae/models.py): VAE architectures and losses

### SegFormer Workflow

- [SegFormer/train.py](SegFormer/train.py): fine-tune SegFormer on CARLA semantic segmentation
- [SegFormer/evaluation.py](SegFormer/evaluation.py): evaluate a trained checkpoint on the validation split
- [SegFormer/inference.py](SegFormer/inference.py): run prediction on a single image
- [SegFormer/inspect_segformer.py](SegFormer/inspect_segformer.py): interactive viewer for RGB, ground truth, prediction, and overlay

## Data Layout

The perception pipelines expect CARLA data arranged like this:

```text
dataset_root/
├── rgb/
└── segmentation/
```

For the current setup in this repo:

- RGB images are stored at `1280x720`
- SegFormer training uses aspect-ratio-preserving resize plus padding to `512x512`
- segmentation masks are converted from CARLA semantic colors into the repo's 13-class label set

## Typical Workflows

### 1. Collect Data

```bash
python CarlaEnv/collect_data.py --output_dir vae/my_data -start_carla
```

### 2. Train a VAE

```bash
cd vae
python train_vae.py --model_name my_trained_vae --dataset my_data
```

Launch the VAE inspector:

```bash
cd vae
python inspect_vae.py --model_dir models/my_trained_vae
```

### 3. Train SegFormer

```bash
python3 -m SegFormer.train --dataset_dir vae/my_data_autopilot --num_workers 8
```

TensorBoard:

```bash
tensorboard --logdir models/segformer/logs
```

Interactive segmentation inspector:

```bash
python3 -m SegFormer.inspect_segformer \
  --checkpoint models/segformer/best_model.pt \
  --dataset_dir vae/my_data_autopilot
```

Validation metrics:

```bash
python3 -m SegFormer.evaluation \
  --checkpoint models/segformer/best_model.pt \
  --dataset_dir vae/my_data_autopilot
```

### 4. Train a PPO Agent

```bash
python train.py --model_name name_of_your_model -start_carla
```

Or use a JSON config:

```bash
python train.py --config config/train.example.json
```

Evaluate a trained PPO agent:

```bash
python run_eval.py --model_name pretrained_agent -start_carla
```

Or use a JSON config:

```bash
python run_eval.py --config config/eval.example.json
```

Run the classical PID controller baseline on the route environment:

```bash
python run_controller.py --config config/controller_pid.example.json
```

Inspect policy behavior:

```bash
python inspect_agent.py --model_name name_of_your_model
```

## Logging and Checkpoints

### SegFormer

- checkpoints: `models/segformer/best_model.pt` and `models/segformer/last_model.pt`
- TensorBoard logs: `models/segformer/logs`
- tracked metrics: loss, mIoU, pixel accuracy, epoch timing, throughput

### VAE

- checkpoints and TensorBoard logs are stored under `vae/models/`

### PPO

- checkpoints, videos, and logs are stored under `models/<model_name>/`

## Config Files

The main entry points now support `--config <path>` with flat JSON config files.

Examples:

- `config/train.example.json`
- `config/eval.example.json`
- `config/autopilot_collector.example.json`
- `config/controller_pid.example.json`
- `config/lab.example.json`

The CARLA Env Lab can also save and load launcher setups as JSON through its `Save Setup` and `Load Setup` buttons.

## Repo Structure

```text
CarlaAI/
├── CarlaEnv/        # CARLA envs, data collection, planners, wrappers
├── SegFormer/       # semantic segmentation training, eval, inference, UI
├── vae/             # VAE training and inspection tools
├── models/          # PPO checkpoints and logs
├── doc/             # write-up and figures
├── train.py         # PPO training entry point
├── run_eval.py      # PPO evaluation entry point
├── inspect_agent.py # PPO policy inspection UI
└── vae_common.py    # VAE/PPO shared state-encoding utilities
```

## Notes on the Current SegFormer Setup

- transfer learning is used rather than training from scratch
- the pretrained backbone comes from Hugging Face SegFormer checkpoints
- the classifier head is adapted from the pretrained label count to the repo's 13 CARLA classes
- preprocessing uses aspect-ratio-preserving resize and padding instead of direct stretching

## Base Repository and Citation

This repository is based on the original `bitsauce/Carla-ppo` implementation. The current branch history includes an import of that codebase and extends it with additional perception workflows, updated datasets, and SegFormer-based semantic segmentation.

Base repository:

- `bitsauce/Carla-ppo`: https://github.com/bitsauce/Carla-ppo

If you use this repo, please also cite or acknowledge the original base repository alongside your modifications.

## Related References

- CARLA simulator: https://carla.org/
- Project write-up in this repo: [doc/Accelerating_Training_of_DeepRL_Based_AV_Agents_Through_Env_Designs.pdf](doc/Accelerating_Training_of_DeepRL_Based_AV_Agents_Through_Env_Designs.pdf)
- Learning to Drive in a Day: https://arxiv.org/abs/1807.00412
- End-to-end Driving via Conditional Imitation Learning: https://arxiv.org/abs/1710.02410

## Status

Currently working in this repo:

- latent-state learning with a VAE
- supervised semantic segmentation with SegFormer
- the original PPO training codebase and CARLA environment stack

## TODO

- test PPO end-to-end against the current codebase state
- integrate PPO cleanly with the newer perception-side changes
- evaluate how SegFormer or segmentation-derived state representations should feed into PPO
