# Phase 1 Bootstrap Plan

This note turns the architecture in [README_Pilot_Project.md](../README_Pilot_Project.md) into an implementation order that reuses the current repository instead of starting over.

## Why Start Here

The current repo already contains the most valuable Phase 1 building blocks:

- `CarlaEnv/carla_lap_env.py`
- `CarlaEnv/carla_route_env.py`
- `CarlaEnv/wrappers.py`
- `scripts/carla_env_lab.py`

These files already cover the native CARLA control loop, sensors, route/lap tasks, and a lightweight local UI. Reusing and restructuring them will get us to a usable MVP much faster than beginning with PPO, VAE, or ROS2 integration.

## Immediate Goal

Create a clean, reusable native CARLA foundation that:

- owns reset/step/render and episode lifecycle in one place
- supports both lap and route tasks through a shared interface
- exposes a stable observation/action contract
- records a common rollout trace for evaluation
- can be launched and inspected through the existing lab UI

## Recommended Order

### 1. Stabilize the Native Environment Contract

Start by refactoring the duplicated logic in:

- `CarlaEnv/carla_lap_env.py`
- `CarlaEnv/carla_route_env.py`

Common responsibilities that should move into a shared base layer:

- CARLA process startup and connection
- pygame and viewer setup
- world, vehicle, camera, and HUD initialization
- synchronous stepping behavior
- observation buffering
- generic reset/close/render plumbing
- shared metric accumulation

End state:

- one shared base env for simulator lifecycle
- small task-specific envs for lap and route behavior

### 2. Define Minimal Core Interfaces

Before adding features, define lightweight internal contracts for:

- `ObservationProvider`
- `ActionInterface`
- `ScenarioConfig`
- `EpisodeMetrics`

This does not need a full plugin system yet. Simple Python classes or dataclasses are enough for v0.1.

End state:

- no more hidden coupling between task logic and env internals
- new controllers can target a stable interface

### 3. Make `scripts/carla_env_lab.py` the Main Smoke-Test Tool

Reuse the existing launcher as the Phase 1 operator UI.

It should become the fastest way to:

- launch lap or route scenarios
- manually drive
- inspect cameras and route overlays
- run env probes
- validate state/action shapes
- later launch controller tests

End state:

- a practical local testing surface without writing throwaway scripts

### 4. Add Shared Rollout Logging and Metrics

Metrics should not stay scattered across training code or task code.

Create one shared rollout record that captures:

- timestamp or step index
- ego pose
- speed
- action applied
- collision and lane events
- route progress
- reward fields where relevant

From that trace, compute common metrics such as:

- route completion
- collisions
- lane deviation
- average speed
- comfort-related signals

End state:

- fair comparison across manual runs, classical controllers, and RL later

### 5. Externalize Configuration

After the env contract is cleaner, move hardcoded values into layered config:

- simulator config
- vehicle config
- sensor rig config
- scenario/task config
- experiment/run config

This can start small. The main value is to stop burying scenario assumptions directly in env constructors.

End state:

- reproducible experiments
- cleaner env constructors
- easier ROS2 bridge alignment later

### 6. Add ROS2 Bridge Integration After the Native Loop Is Stable

Do not begin by pushing ROS2 into the execution-critical path.

Instead:

- keep the wrapper as the authoritative runtime path
- mirror the already-working simulator state into ROS2
- use ROS2 first for visualization, inspection, and optional controller adapters

End state:

- ROS2 adds visibility without destabilizing the fast loop

## What Not To Start With

Avoid making these the first milestone:

- PPO cleanup
- VAE integration
- SegFormer integration
- full plugin architecture
- complex ROS2 control loops

Those should plug into the Phase 1 foundation after the native env contract is cleaned up.

## First Refactor Slice

The first coding milestone should stay small:

1. Extract shared env lifecycle code from `carla_lap_env.py` and `carla_route_env.py`.
2. Introduce a shared base env module.
3. Keep lap and route behavior in thin task-specific env classes.
4. Preserve current behavior so the lab UI still works.
5. Add or repair smoke tests around reset, step, and observation/action shapes.

If this slice lands cleanly, the repo will already have a much better foundation for Phase 1 without blocking current experiments.

## Suggested Next Implementation Target

Create a new shared env module under `CarlaEnv/` and move duplicated setup code there first. That gives the biggest architectural win with the lowest product risk.
