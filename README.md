# CarlaAI

## Short Project Summary

CarlaAI is a lightweight CARLA-based autonomous driving research repository for experimenting with both classical controllers and learning-based methods in a shared environment stack.

The repo currently brings together:

- CARLA route/lap environments with visualization and debugging support
- classical controller baselines such as PID and kinematic MPC
- PPO policy, training, evaluation, and inspection code
- VAE-based representation learning
- SegFormer-based semantic segmentation tooling
- data collection and inspection utilities

The longer-term goal is to study not only controller performance, but also how user comfort, trust, and preference-aware constraints can shape autonomous driving behavior.


## What the Repo Currently Provides

- Gym-like CARLA environments for route-following and lap-following tasks
- PID and modular kinematic MPC controller baselines
- PPO training, evaluation, and policy-inspection entry points
- VAE training and reconstruction inspection tools
- SegFormer training, evaluation, inference, and inspection tools
- Route/HUD/debug overlays and controller trace logging
- JSON-config-driven experiment entry points

## Main Workflows

### Run a Classical Controller

PID baseline:

```bash
python3 -m scripts.run_controller --config config/controller_pid.example.json
```

Kinematic MPC baseline:

```bash
python3 -m scripts.run_controller --config config/controller_mpc.example.json
```

Analyze an MPC debug trace:

```bash
python3 -m scripts.analyze_mpc_run models/controller_runs/mpc_route_debug.json
```

### Train and Evaluate PPO

Train:

```bash
python3 -m scripts.ppo_train --config config/train.example.json
```

Evaluate:

```bash
python3 -m scripts.ppo_eval --config config/eval.example.json
```

Inspect a trained policy:

```bash
python3 -m scripts.ppo_inspect --model_name name_of_your_model
```

### Collect Data

```bash
python3 CarlaEnv/collect_data.py --output_dir perception/vae/my_data -start_carla
```

### Train a VAE

```bash
python3 perception/vae/train_vae.py --model_name my_trained_vae --dataset my_data
```

Inspect reconstructions:

```bash
python3 perception/vae/inspect_vae.py --model_dir models/my_trained_vae
```

### Train SegFormer

```bash
python3 -m perception.segformer.train --dataset_dir perception/vae/my_data_autopilot --num_workers 8
```

Inspect predictions:

```bash
python3 -m perception.segformer.inspect_segformer \
  --checkpoint models/segformer/best_model.pt \
  --dataset_dir perception/vae/my_data_autopilot
```

Evaluate:

```bash
python3 -m perception.segformer.evaluation \
  --checkpoint models/segformer/best_model.pt \
  --dataset_dir perception/vae/my_data_autopilot
```

## Repo Structure

```text
CarlaAI/
├── CarlaEnv/          # CARLA environments, planners, HUD, wrappers, and simulator glue
├── control/           # classical control modules
│   ├── mpc/           # modular kinematic MPC implementation
│   └── pid/           # baseline-facing PID runner/planner adapter
├── perception/        # perception models, datasets, and shared helpers
│   ├── segformer/     # semantic segmentation workflow
│   ├── vae/           # VAE workflow
│   └── common/        # shared perception helpers
├── learned_policies/  # learned driving policies
│   └── rl/ppo/        # PPO policy network, train/eval logic, rewards, and helpers
├── config/            # example JSON configs
├── docs/              # notes, write-up, figures
├── models/            # checkpoints and logs
├── scripts/           # runnable entry points
│   ├── ppo_train.py
│   ├── ppo_eval.py
│   ├── ppo_inspect.py
│   ├── run_controller.py
│   ├── analyze_mpc_run.py
│   └── carla_env_lab.py
```

## Layout Notes

- `CarlaEnv/` remains the runtime backbone for CARLA-specific environment and simulator integration.
- `control/` contains the classical controller implementations that sit on top of the environment stack.
- `perception/` contains representation and segmentation models plus shared adapters used by training/evaluation code.
- `learned_policies/rl/ppo/` contains the PPO implementation itself; `scripts/` provides the stable CLI entrypoints.

## Config / Examples

Main entry points support `--config <path>` with flat JSON config files.

Examples:

- `config/train.example.json`
- `config/eval.example.json`
- `config/autopilot_collector.example.json`
- `config/controller_pid.example.json`
- `config/controller_mpc.example.json`
- `config/lab.example.json`

The controller flow uses shared defaults from [CarlaEnv/config.py](CarlaEnv/config.py), so controller runtime settings such as target speed, MPC horizon, and MPC `dt` can be managed centrally and overridden when needed.

## TODO

- implement a dynamic-model MPC baseline
- add obstacle-aware MPC constraints and avoidance logic
- integrate PPO more cleanly with the newer environment/controller baseline
- improve shared metrics for comparing PID, MPC, and PPO runs
- continue extending the repo toward a lightweight hybrid experimentation framework
