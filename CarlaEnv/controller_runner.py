import json
import os

import pygame
from pygame.locals import K_TAB, QUIT

try:
    from .agents.navigation.local_planner import LocalPlanner
    from .manual_control import _apply_manual_key_state, _handle_manual_control_event
    from .mpc_controller import KinematicMPCController, MPCConfig, build_vehicle_state
except ImportError:
    from agents.navigation.local_planner import LocalPlanner
    from manual_control import _apply_manual_key_state, _handle_manual_control_event
    from mpc_controller import KinematicMPCController, MPCConfig, build_vehicle_state


def _create_pid_local_planner(vehicle, target_speed, fps):
    dt = 1.0 / fps
    args_lateral = {"K_P": 1.0, "K_D": 0.02, "K_I": 0.0, "dt": dt}
    args_longitudinal = {"K_P": 1.0, "K_D": 0.1, "K_I": 0.0, "dt": dt}
    return LocalPlanner(
        vehicle,
        opt_dict={
            "target_speed": target_speed,
            "lateral_control_dict": args_lateral,
            "longitudinal_control_dict": args_longitudinal,
        },
    )


def _handle_controller_events(control, manual_override):
    toggle_manual_override = False
    should_close = False
    for event in pygame.event.get():
        if event.type == QUIT:
            should_close = True
            break
        if event.type == pygame.KEYDOWN and event.key == K_TAB:
            toggle_manual_override = True
            continue
        if manual_override and _handle_manual_control_event(event, control):
            should_close = True
            break
    return manual_override ^ toggle_manual_override, should_close, toggle_manual_override


def _episode_output_path(base_path, episode_idx, suffix):
    root, ext = os.path.splitext(base_path)
    ext = ext or ".json"
    return f"{root}_episode_{episode_idx:03d}_{suffix}{ext}"


def run_pid_controller(env, target_speed=20.0, max_episodes=1, render=True, summary_path=None):
    episode_summaries = []

    for episode_idx in range(max_episodes):
        state = env.reset(is_training=False)
        del state  # controller does not consume encoded observations directly

        planner = _create_pid_local_planner(env.vehicle, target_speed=target_speed, fps=env.fps)
        planner.set_global_plan(env.route_waypoints)
        current_route_identity = id(env.route_waypoints)
        manual_override = False

        terminal = False
        info = {"closed": False}
        while not terminal:
            manual_override, should_close, toggled = _handle_controller_events(env.vehicle.control, manual_override)
            if should_close:
                env.close()
                break

            if id(env.route_waypoints) != current_route_identity:
                planner.set_global_plan(env.route_waypoints)
                current_route_identity = id(env.route_waypoints)

            if toggled and not manual_override:
                planner = _create_pid_local_planner(env.vehicle, target_speed=target_speed, fps=env.fps)
                planner.set_global_plan(env.route_waypoints)
                current_route_identity = id(env.route_waypoints)

            if manual_override:
                _apply_manual_key_state(env.vehicle.control)
            else:
                control = planner.run_step(debug=False)
                env.vehicle.control = control

            _, _, terminal, info = env.step(None)

            env.extra_info.extend(
                [
                    f"Episode {episode_idx}",
                    "Controller: PID Local Planner",
                    f"Mode: {'Manual Override' if manual_override else 'Automatic'} (Tab to toggle)",
                    f"Target speed: {target_speed:.1f} km/h",
                    "",
                ]
            )
            if render and not info["closed"]:
                env.render()

            if info["closed"]:
                break

        summary = env.get_episode_summary()
        summary["controller_type"] = "pid_local_planner"
        summary["target_speed_kmh"] = target_speed
        summary["episode_idx"] = episode_idx
        episode_summaries.append(summary)

        if info.get("closed"):
            break

    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": episode_summaries}, f, indent=2)

    return episode_summaries


def run_mpc_controller(
    env,
    target_speed=None,
    max_episodes=1,
    render=True,
    summary_path=None,
    debug_trace_path=None,
    horizon=None,
    dt=None,
    config=None,
):
    episode_summaries = []
    mpc_config = config or MPCConfig()
    if horizon is not None:
        mpc_config.horizon = int(horizon)
    if dt is not None:
        mpc_config.dt = float(dt)
    if target_speed is None:
        target_speed = 20.0

    for episode_idx in range(max_episodes):
        state = env.reset(is_training=False)
        del state

        controller = KinematicMPCController(
            target_speed_kmh=target_speed,
            config=mpc_config,
        )
        controller.reset()
        manual_override = False
        debug_steps = []

        terminal = False
        info = {"closed": False}
        while not terminal:
            manual_override, should_close, toggled = _handle_controller_events(env.vehicle.control, manual_override)
            if should_close:
                env.close()
                break

            if toggled and not manual_override:
                controller.reset()

            if manual_override:
                _apply_manual_key_state(env.vehicle.control)
                debug = None
            else:
                vehicle_state = build_vehicle_state(env.vehicle)
                steer, accel, debug = controller.run_step(
                    vehicle_state,
                    env.route_waypoints,
                    env.current_waypoint_index,
                )

                env.vehicle.control.steer = float(steer)
                if accel >= 0.0:
                    env.vehicle.control.throttle = float(min(1.0, accel / max(controller.config.max_accel, 1e-6)))
                    env.vehicle.control.brake = 0.0
                else:
                    env.vehicle.control.throttle = 0.0
                    env.vehicle.control.brake = float(min(1.0, (-accel) / max(controller.config.max_decel, 1e-6)))

            _, _, terminal, info = env.step(None)

            step_debug = {
                "step": int(env.step_count),
                "manual_override": bool(manual_override),
                "vehicle": {
                    "x": float(env.vehicle.get_transform().location.x),
                    "y": float(env.vehicle.get_transform().location.y),
                    "yaw_deg": float(env.vehicle.get_transform().rotation.yaw),
                    "speed_kmh": float(3.6 * env.vehicle.get_speed()),
                },
                "control": {
                    "steer": float(env.vehicle.control.steer),
                    "throttle": float(env.vehicle.control.throttle),
                    "brake": float(env.vehicle.control.brake),
                },
                "tracking": {
                    "distance_from_center": float(getattr(env, "distance_from_center", 0.0)),
                    "current_waypoint_index": int(getattr(env, "current_waypoint_index", 0)),
                },
            }
            if debug is not None:
                step_debug["mpc"] = controller.get_debug_snapshot()
            debug_steps.append(step_debug)

            env.extra_info.extend(
                [
                    f"Episode {episode_idx}",
                    "Controller: Kinematic MPC",
                    f"Mode: {'Manual Override' if manual_override else 'Automatic'} (Tab to toggle)",
                    f"Target speed: {target_speed:.1f} km/h",
                    f"MPC horizon: {controller.config.horizon} | dt: {controller.config.dt:.2f}s",
                    f"Solver: {'OK' if debug.solver_success else 'Fallback'} | iters: {debug.iterations}" if not manual_override else "Solver: paused during manual override",
                    "",
                ]
            )
            if render and not info["closed"]:
                env.render()

            if info["closed"]:
                break

        summary = env.get_episode_summary()
        summary["controller_type"] = "kinematic_mpc"
        summary["target_speed_kmh"] = target_speed
        summary["episode_idx"] = episode_idx
        summary["mpc_horizon"] = int(controller.config.horizon)
        summary["mpc_dt"] = float(controller.config.dt)
        episode_summaries.append(summary)

        episode_payload = {
            "controller_type": "kinematic_mpc",
            "target_speed_kmh": target_speed,
            "mpc_horizon": int(controller.config.horizon),
            "mpc_dt": float(controller.config.dt),
            "episode_idx": int(episode_idx),
            "summary": summary,
            "steps": debug_steps,
        }

        if summary_path:
            episode_summary_path = _episode_output_path(summary_path, episode_idx, "summary")
            os.makedirs(os.path.dirname(episode_summary_path) or ".", exist_ok=True)
            with open(episode_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        if debug_trace_path:
            episode_debug_path = _episode_output_path(debug_trace_path, episode_idx, "debug")
            os.makedirs(os.path.dirname(episode_debug_path) or ".", exist_ok=True)
            with open(episode_debug_path, "w", encoding="utf-8") as f:
                json.dump(episode_payload, f, indent=2)

        if info.get("closed"):
            break

    if summary_path:
        summary_index_path = os.path.splitext(summary_path)[0] + "_index.json"
        with open(summary_index_path, "w", encoding="utf-8") as f:
            json.dump({"episodes": episode_summaries}, f, indent=2)

    if debug_trace_path:
        debug_index_path = os.path.splitext(debug_trace_path)[0] + "_index.json"
        with open(debug_index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "controller_type": "kinematic_mpc",
                    "target_speed_kmh": target_speed,
                    "mpc_horizon": int(controller.config.horizon),
                    "mpc_dt": float(controller.config.dt),
                    "num_episodes": len(episode_summaries),
                },
                f,
                indent=2,
            )

    return episode_summaries
