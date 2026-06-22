import argparse

from CarlaEnv.config import ControllerConfig, namespace_to_controller_config, namespace_to_env_config, parse_args_with_config
from CarlaEnv.controller_runner import run_mpc_controller, run_pid_controller

USE_ROUTE_ENVIRONMENT = True

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


def main():
    controller_defaults = ControllerConfig()
    parser = argparse.ArgumentParser(description="Run a classical controller baseline in CARLA")
    parser.add_argument("--controller", type=str, default="pid", choices=["pid", "mpc"], help="Controller baseline to run")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--viewer_res", type=str, default="1280x720", help="Viewer resolution")
    parser.add_argument("--obs_res", type=str, default="160x80", help="Observation resolution")
    parser.add_argument("--show_waypoints", type=int, default=1, help="Render waypoint overlay")
    parser.add_argument("--synchronous", type=int, default=1, help="Synchronous mode (0/1)")
    parser.add_argument("--fps", type=int, default=30, help="Simulation FPS")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CARLA")
    parser.add_argument("--target_speed", type=float, default=controller_defaults.target_speed, help="Controller target speed in km/h")
    parser.add_argument("--episodes", type=int, default=controller_defaults.episodes, help="Number of episodes to run")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional JSON path to write episode summaries")
    parser.add_argument("--debug_trace_path", type=str, default=None, help="Optional JSON path to write per-step MPC debug traces")
    parser.add_argument("--mpc_horizon", type=int, default=controller_defaults.mpc_horizon, help="MPC prediction horizon")
    parser.add_argument("--mpc_dt", type=float, default=controller_defaults.mpc_dt, help="MPC discretization step in seconds")

    args = parse_args_with_config(parser)
    simulator_config, display_config = namespace_to_env_config(args)
    controller_config = namespace_to_controller_config(args)

    env = CarlaEnv(
        host=simulator_config.host,
        port=simulator_config.port,
        viewer_res=display_config.viewer_res,
        obs_res=display_config.obs_res,
        synchronous=simulator_config.synchronous,
        fps=simulator_config.fps,
        start_carla=simulator_config.start_carla,
        show_waypoints=display_config.show_waypoints,
    )

    try:
        if args.controller == "pid":
            summaries = run_pid_controller(
                env,
                target_speed=controller_config.target_speed,
                max_episodes=controller_config.episodes,
                render=True,
                summary_path=args.summary_path,
            )
        else:
            summaries = run_mpc_controller(
                env,
                target_speed=controller_config.target_speed,
                max_episodes=controller_config.episodes,
                render=True,
                summary_path=args.summary_path,
                debug_trace_path=args.debug_trace_path,
                horizon=controller_config.mpc_horizon,
                dt=controller_config.mpc_dt,
            )
        for summary in summaries:
            print(summary)
    finally:
        env.close()


if __name__ == "__main__":
    main()
