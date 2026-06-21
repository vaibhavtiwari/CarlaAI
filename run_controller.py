import argparse

from CarlaEnv.config import namespace_to_env_config, parse_args_with_config
from CarlaEnv.controller_runner import run_pid_controller

USE_ROUTE_ENVIRONMENT = True

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


def main():
    parser = argparse.ArgumentParser(description="Run a classical controller baseline in CARLA")
    parser.add_argument("--controller", type=str, default="pid", choices=["pid"], help="Controller baseline to run")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--viewer_res", type=str, default="1280x720", help="Viewer resolution")
    parser.add_argument("--obs_res", type=str, default="160x80", help="Observation resolution")
    parser.add_argument("--show_waypoints", type=int, default=1, help="Render waypoint overlay")
    parser.add_argument("--synchronous", type=int, default=1, help="Synchronous mode (0/1)")
    parser.add_argument("--fps", type=int, default=30, help="Simulation FPS")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CARLA")
    parser.add_argument("--target_speed", type=float, default=20.0, help="Controller target speed in km/h")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional JSON path to write episode summaries")

    args = parse_args_with_config(parser)
    simulator_config, display_config = namespace_to_env_config(args)

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
        summaries = run_pid_controller(
            env,
            target_speed=args.target_speed,
            max_episodes=args.episodes,
            render=True,
            summary_path=args.summary_path,
        )
        for summary in summaries:
            print(summary)
    finally:
        env.close()


if __name__ == "__main__":
    main()
