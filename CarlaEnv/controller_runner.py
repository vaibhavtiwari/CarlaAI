import json
import time

try:
    from .agents.navigation.local_planner import LocalPlanner
except ImportError:
    from agents.navigation.local_planner import LocalPlanner


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


def run_pid_controller(env, target_speed=20.0, max_episodes=1, render=True, summary_path=None):
    episode_summaries = []

    for episode_idx in range(max_episodes):
        state = env.reset(is_training=False)
        del state  # controller does not consume encoded observations directly

        planner = _create_pid_local_planner(env.vehicle, target_speed=target_speed, fps=env.fps)
        planner.set_global_plan(env.route_waypoints)
        current_route_identity = id(env.route_waypoints)

        terminal = False
        while not terminal:
            if id(env.route_waypoints) != current_route_identity:
                planner.set_global_plan(env.route_waypoints)
                current_route_identity = id(env.route_waypoints)

            control = planner.run_step(debug=False)
            env.vehicle.control = control
            _, _, terminal, info = env.step(None)

            env.extra_info.extend(
                [
                    f"Episode {episode_idx}",
                    "Controller: PID Local Planner",
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
