import os
import random
import shutil

import numpy as np
import torch

from CarlaEnv.config import namespace_to_env_config, parse_args_with_config
from learned_policies.rl.ppo import PPO
from perception.common.segformer_common import create_encode_state_fn, load_segformer
from .reward_functions import reward_functions
from .run_eval import run_eval
from .utils import compute_gae

USE_ROUTE_ENVIRONMENT = False

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


def train(params, start_carla=True, restart=False):
    # Read parameters
    learning_rate    = params["learning_rate"]
    lr_decay         = params["lr_decay"]
    discount_factor  = params["discount_factor"]
    gae_lambda       = params["gae_lambda"]
    ppo_epsilon      = params["ppo_epsilon"]
    initial_std      = params["initial_std"]
    value_scale      = params["value_scale"]
    entropy_scale    = params["entropy_scale"]
    horizon          = params["horizon"]
    num_epochs       = params["num_epochs"]
    num_episodes     = params["num_episodes"]
    batch_size       = params["batch_size"]
    segformer_checkpoint = params["segformer_checkpoint"]
    segformer_backbone   = params["segformer_backbone"]
    segformer_width      = params["segformer_width"]
    segformer_height     = params["segformer_height"]
    segformer_num_classes = params["segformer_num_classes"]
    device           = params["device"]
    synchronous      = params["synchronous"]
    fps              = params["fps"]
    action_smoothing = params["action_smoothing"]
    model_name       = params["model_name"]
    reward_fn        = params["reward_fn"]
    seed             = params["seed"]
    eval_interval    = params["eval_interval"]
    record_eval      = params["record_eval"]

    # Set seeds
    if isinstance(seed, int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Load SegFormer encoder
    segformer = load_segformer(
        checkpoint_path=segformer_checkpoint,
        pretrained_model_name=segformer_backbone,
        image_size=(segformer_width, segformer_height),
        num_classes=segformer_num_classes,
        device=device,
    )

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed"])
    encode_state_fn = create_encode_state_fn(segformer, measurements_to_include)

    # Create env
    print("Creating environment")
    env = CarlaEnv(host=params.get("host", "127.0.0.1"),
                   port=params.get("port", 2000),
                   viewer_res=params.get("viewer_res", (1280, 720)),
                   obs_res=params.get("obs_res", (160, 80)),
                   show_waypoints=params.get("show_waypoints", True),
                   action_smoothing=action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn=reward_functions[reward_fn],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=start_carla)
    if isinstance(seed, int):
        env.seed(seed)
    best_eval_reward = -float("inf")

    # Environment constants
    input_shape = np.array([segformer.feature_dim + len(measurements_to_include)])
    num_actions = env.action_space.shape[0]

    # Create model
    print("Creating model")
    model = PPO(input_shape, env.action_space,
                learning_rate=learning_rate, lr_decay=lr_decay,
                epsilon=ppo_epsilon, initial_std=initial_std,
                value_scale=value_scale, entropy_scale=entropy_scale,
                model_dir=os.path.join("models", model_name),
                device=device)

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
            if answer.upper() == "C":
                pass
            elif answer.upper() == "R":
                restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(model_name))
    
    if restart:
        shutil.rmtree(model.model_dir)
        for d in model.dirs:
            os.makedirs(d)
    model.init_session()
    if not restart:
        model.load_latest_checkpoint()
    model.write_dict_to_summary("hyperparameters", params, 0)

    # For every episode
    while num_episodes <= 0 or model.get_episode_idx() < num_episodes:
        episode_idx = model.get_episode_idx()
        
        # Run evaluation periodically
        if episode_idx % eval_interval == 0:
            video_filename = None
            if record_eval:
                video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
            eval_reward = run_eval(env, model, video_filename=video_filename)
            eval_summary = env.get_episode_summary()
            model.write_value_to_summary("eval/reward", eval_reward, episode_idx)
            model.write_value_to_summary("eval/distance_traveled", eval_summary["distance_traveled"], episode_idx)
            model.write_value_to_summary("eval/average_speed", eval_summary["average_speed_kmh"], episode_idx)
            model.write_value_to_summary("eval/center_lane_deviation", eval_summary["center_lane_deviation"], episode_idx)
            model.write_value_to_summary("eval/average_center_lane_deviation", eval_summary["average_center_lane_deviation"], episode_idx)
            model.write_value_to_summary("eval/distance_over_deviation", eval_summary["distance_over_deviation"], episode_idx)
            model.write_value_to_summary("eval/collisions", eval_summary["collisions"], episode_idx)
            model.write_value_to_summary("eval/lane_invasions", eval_summary["lane_invasions"], episode_idx)
            if eval_reward > best_eval_reward:
                model.save()
                best_eval_reward = eval_reward

        # Reset environment
        state, terminal_state, total_reward = env.reset(), False, 0
        
        # While episode not done
        print(f"Episode {episode_idx} (Step {model.get_train_step_idx()})")
        while not terminal_state:
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            for _ in range(horizon):
                action, value = model.predict(state, write_to_summary=True)

                # Perform action
                new_state, reward, terminal_state, info = env.step(action)

                if info["closed"] == True:
                    exit(0)
                    
                env.extra_info.extend([
                    "Episode {}".format(episode_idx),
                    "Training...",
                    "",
                    "Value:  % 20.2f" % value
                ])

                env.render()
                total_reward += reward

                # Store state, action and reward
                states.append(state)         # [T, *input_shape]
                taken_actions.append(action) # [T,  num_actions]
                values.append(value)         # [T]
                rewards.append(reward)       # [T]
                dones.append(terminal_state) # [T]
                state = new_state

                if terminal_state:
                    break

            # Calculate last value (bootstrap value)
            _, last_values = model.predict(state) # []
            
            # Compute GAE
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten arrays
            states        = np.array(states)
            taken_actions = np.array(taken_actions)
            returns       = np.array(returns)
            advantages    = np.array(advantages)

            T = len(rewards)
            assert states.shape == (T, *input_shape)
            assert taken_actions.shape == (T, num_actions)
            assert returns.shape == (T,)
            assert advantages.shape == (T,)

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / batch_size))):
                    # Sample mini-batch randomly
                    begin = i * batch_size
                    end   = begin + batch_size
                    if end > num_samples:
                        end = None
                    mb_idx = indices[begin:end]

                    # Optimize network
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])

        # Write episodic values
        train_summary = env.get_episode_summary()
        model.write_value_to_summary("train/reward", total_reward, episode_idx)
        model.write_value_to_summary("train/distance_traveled", train_summary["distance_traveled"], episode_idx)
        model.write_value_to_summary("train/average_speed", train_summary["average_speed_kmh"], episode_idx)
        model.write_value_to_summary("train/center_lane_deviation", train_summary["center_lane_deviation"], episode_idx)
        model.write_value_to_summary("train/average_center_lane_deviation", train_summary["average_center_lane_deviation"], episode_idx)
        model.write_value_to_summary("train/distance_over_deviation", train_summary["distance_over_deviation"], episode_idx)
        model.write_value_to_summary("train/collisions", train_summary["collisions"], episode_idx)
        model.write_value_to_summary("train/lane_invasions", train_summary["lane_invasions"], episode_idx)
        model.write_episodic_summaries()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trains a CARLA agent with PPO")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0, help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")
    parser.add_argument("--horizon", type=int, default=128, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=0, help="Number of episodes to train for (0 or less trains forever)")

    # SegFormer parameters
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

    # Training parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See learned_policies/rl/ppo/reward_functions.py for more info.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed " +
                             "with this option in our experience)")
    parser.add_argument("--eval_interval", type=int, default=5, help="Number of episodes between evaluation runs")
    parser.add_argument("--record_eval", type=bool, default=True,
                        help="If True, save videos of evaluation episodes " +
                             "to models/model_name/videos/")
    
    parser.add_argument("-restart", action="store_true",
                        help="If True, delete existing model in models/model_name before starting training")

    args = parse_args_with_config(parser)
    simulator_config, display_config = namespace_to_env_config(args)
    params = vars(args)
    params["host"] = simulator_config.host
    params["port"] = simulator_config.port
    params["synchronous"] = simulator_config.synchronous
    params["fps"] = simulator_config.fps
    params["start_carla"] = simulator_config.start_carla
    params["viewer_res"] = display_config.viewer_res
    params["obs_res"] = display_config.obs_res
    params["show_waypoints"] = display_config.show_waypoints

    # Remove a couple of parameters that we dont want to log
    start_carla = params["start_carla"]; del params["start_carla"]
    restart = params["restart"]; del params["restart"]

    # Start training
    train(params, start_carla, restart)


if __name__ == "__main__":
    main()
