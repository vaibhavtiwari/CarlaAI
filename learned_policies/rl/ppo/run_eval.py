import os
import random
import re
import shutil
import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import transform

try:
    import gym
except ImportError:
    import gymnasium as gym

from CarlaEnv.config import namespace_to_env_config, parse_args_with_config
from learned_policies.rl.ppo import PPO
from CarlaEnv.wrappers import angle_diff, vector
from .utils import VideoRecorder, compute_gae
from perception.common.segformer_common import create_encode_state_fn, load_segformer
from .reward_functions import reward_functions

USE_ROUTE_ENVIRONMENT = False

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


def run_eval(env, model, video_filename=None):
    # Init test env
    state, terminal, total_reward = env.reset(is_training=False), False, 0
    rendered_frame = env.render(mode="rgb_array")

    # Init video recording
    if video_filename is not None:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, int(env.average_fps)))
        video_recorder = VideoRecorder(video_filename,
                                       frame_size=rendered_frame.shape,
                                       fps=env.average_fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = model.get_episode_idx()

    # While non-terminal state
    while not terminal:
        env.extra_info.append("Episode {}".format(episode_idx))
        env.extra_info.append("Running eval...".format(episode_idx))
        env.extra_info.append("")

        # Take deterministic actions at test time (std=0)
        action, _ = model.predict(state, greedy=True)
        state, reward, terminal, info = env.step(action)

        if info["closed"] == True:
            break

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if video_recorder is not None:
            video_recorder.add_frame(rendered_frame)
        total_reward += reward

    # Release video
    if video_recorder is not None:
        video_recorder.release()

    if info["closed"] == True:
        exit(0)

    return total_reward

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runs the model in evaluation mode")
    
    # Model params
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See learned_policies/rl/ppo/reward_functions.py for more info.")
    parser.add_argument("--segformer_checkpoint", type=str, default="models/segformer/best_model.pt",
                        help="Path to a trained SegFormer checkpoint")
    parser.add_argument("--segformer_backbone", type=str,
                        default="nvidia/segformer-b2-finetuned-ade-512-512",
                        help="Pretrained Hugging Face SegFormer backbone name")
    parser.add_argument("--segformer_width", type=int, default=512, help="SegFormer preprocessing width")
    parser.add_argument("--segformer_height", type=int, default=512, help="SegFormer preprocessing height")
    parser.add_argument("--segformer_num_classes", type=int, default=13, help="Number of segmentation classes")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for PPO and SegFormer")

    # Environment settings
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--viewer_res", type=str, default="1280x720", help="Viewer resolution")
    parser.add_argument("--obs_res", type=str, default="160x80", help="Observation resolution")
    parser.add_argument("--show_waypoints", type=int, default=1, help="Render waypoint overlay")
    parser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
    parser.add_argument("--fps", type=int, default=30, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")

    # Recording    
    parser.add_argument("--record_to_file", type=str, default=None, help="File to record evaluation video to (outputs in .avi format)")

    args = parse_args_with_config(parser)
    simulator_config, display_config = namespace_to_env_config(args)

    # Load SegFormer encoder
    segformer = load_segformer(
        checkpoint_path=args.segformer_checkpoint,
        pretrained_model_name=args.segformer_backbone,
        image_size=(args.segformer_width, args.segformer_height),
        num_classes=args.segformer_num_classes,
        device=args.device,
    )

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed"])
    encode_state_fn = create_encode_state_fn(segformer, measurements_to_include)

    # Create env
    print("Creating environment...")
    env = CarlaEnv(host=simulator_config.host,
                   port=simulator_config.port,
                   viewer_res=display_config.viewer_res,
                   obs_res=display_config.obs_res,
                   show_waypoints=display_config.show_waypoints,
                   action_smoothing=args.action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn=reward_functions[args.reward_fn],
                   synchronous=simulator_config.synchronous,
                   fps=simulator_config.fps,
                   start_carla=simulator_config.start_carla)


    # Set seeds
    seed = 0
    if isinstance(seed, int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env.seed(seed)

    # Create model
    print("Creating model...")
    input_shape = np.array([segformer.feature_dim + len(measurements_to_include)])
    model = PPO(input_shape, env.action_space,
                model_dir=os.path.join("models", args.model_name),
                device=args.device)
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    # Run eval
    print("Running eval...")
    run_eval(env, model, video_filename=args.record_to_file)

    # Close env
    print("Done!")
    env.close()


if __name__ == "__main__":
    main()
