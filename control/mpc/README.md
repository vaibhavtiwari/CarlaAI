# MPC Module

This folder contains the modular implementation of the route-tracking MPC controller used in this repo.

## Goal

The purpose of this module is to keep the MPC implementation easy to study and extend.

It is organized so you can learn the controller in layers:

1. vehicle state and controller configuration
2. reference trajectory generation
3. dynamics and optimization problem construction
4. controller execution and debug bookkeeping

## File Structure

- `common.py`
  - shared dataclasses and utility helpers
  - contains:
    - `MPCConfig`
    - `MPCDebug`
    - `VehicleState`
    - `build_vehicle_state(...)`
    - `wrap_angle_numpy(...)`

- `reference.py`
  - builds the reference trajectory from CARLA route waypoints
  - converts the current route progress into a fixed MPC horizon target

- `solver.py`
  - defines the CasADi optimization problem
  - contains:
    - state definition
    - control definition
    - kinematic bicycle dynamics
    - hard constraints
    - optimization bounds

- `controller.py`
  - high-level MPC controller logic
  - handles:
    - warm start state
    - per-step solve calls
    - debug/cost breakdown tracking
    - output control extraction

- `__init__.py`
  - package exports

## Current State/Control Model

State:

- `x`
- `y`
- `yaw`
- `speed`

Control:

- normalized steering command
- longitudinal acceleration

## Current Hard Constraints

The current solver enforces:

- vehicle dynamics
- steering bounds
- acceleration / deceleration bounds
- minimum speed
- maximum speed

## Current Cost Terms

The controller currently penalizes:

- position tracking error
- heading tracking error
- speed tracking error
- steering effort
- acceleration effort
- steering rate change
- acceleration rate change
- terminal state tracking error

These weights live in `MPCConfig`.

## Data Flow

The flow for one MPC step is:

1. `build_vehicle_state(...)` reads the current CARLA ego state.
2. `reference.py` builds a local reference trajectory from the route.
3. `controller.py` sends the current state and reference to the CasADi solver.
4. `solver.py` solves the optimization problem.
5. `controller.py` extracts the first control action and stores debug data.

## How It Connects To The Rest Of The Repo

The main repo still imports through:

- `CarlaEnv/mpc_controller.py`

That file now serves as a compatibility bridge into this folder.

The actual runtime use happens through:

- `CarlaEnv/controller_runner.py`
- `scripts/run_controller.py`
- `scripts/carla_env_lab.py`

## Recommended Learning Order

If you want to learn the implementation step by step, read in this order:

1. `common.py`
2. `reference.py`
3. `solver.py`
4. `controller.py`

That order maps well to the usual MPC concepts:

1. state and parameters
2. reference generation
3. model and constraints
4. solve-and-apply loop

## Good Next Extensions

Natural next improvements for this module are:

- separate cost construction into its own file
- separate hard constraints and soft constraints more explicitly
- add obstacle constraints
- add slack variables for soft safety constraints
- add live parameter tuning hooks
- log predicted state/control horizon for deeper analysis
