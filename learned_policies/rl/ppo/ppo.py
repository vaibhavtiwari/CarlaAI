import glob
import os

import numpy as np
import torch
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _build_mlp(input_dim, hidden_sizes, output_dim, activation=nn.ReLU, output_activation=None):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        action_space,
        initial_std=0.4,
        pi_hidden_sizes=(500, 300),
        vf_hidden_sizes=(500, 300),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(action_space.shape[0])

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

        self.policy_body = _build_mlp(self.input_dim, pi_hidden_sizes, pi_hidden_sizes[-1], activation=nn.ReLU)
        self.action_mean_head = nn.Linear(pi_hidden_sizes[-1], self.num_actions)
        self.action_logstd = nn.Parameter(
            torch.full((self.num_actions,), float(np.log(initial_std)), dtype=torch.float32)
        )

        if vf_hidden_sizes is None:
            self.value_body = self.policy_body
            self.value_head = nn.Linear(pi_hidden_sizes[-1], 1)
        else:
            self.value_body = _build_mlp(self.input_dim, vf_hidden_sizes, vf_hidden_sizes[-1], activation=nn.ReLU)
            self.value_head = nn.Linear(vf_hidden_sizes[-1], 1)

    def _scaled_action_mean(self, inputs):
        features = self.policy_body(inputs)
        action_mean = torch.tanh(self.action_mean_head(features))
        return self.action_low + ((action_mean + 1.0) / 2.0) * (self.action_high - self.action_low)

    def forward(self, inputs):
        action_mean = self._scaled_action_mean(inputs)
        value_features = self.value_body(inputs)
        value = self.value_head(value_features).squeeze(-1)
        std = torch.exp(self.action_logstd).expand_as(action_mean)
        distribution = torch.distributions.Normal(action_mean, std)
        return distribution, value, action_mean


class PPO:
    def __init__(
        self,
        input_shape,
        action_space,
        learning_rate=3e-4,
        lr_decay=0.998,
        epsilon=0.2,
        value_scale=0.5,
        entropy_scale=0.01,
        initial_std=0.4,
        model_dir="./",
        device=None,
    ):
        input_shape = tuple(int(x) for x in np.asarray(input_shape).reshape(-1))
        if len(input_shape) != 1:
            raise ValueError(f"PPO expects a flat input shape, got {input_shape}")

        self.input_dim = input_shape[0]
        self.action_space = action_space
        self.epsilon = epsilon
        self.value_scale = value_scale
        self.entropy_scale = entropy_scale
        self.initial_learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy = PolicyNetwork(self.input_dim, action_space, initial_std=initial_std).to(self.device)
        self.policy_old = PolicyNetwork(self.input_dim, action_space, initial_std=initial_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.train_step_idx = 0
        self.predict_step_idx = 0
        self.episode_idx = 0
        self._episodic_metric_totals = {}
        self._episodic_metric_counts = {}
        self._logging_enabled = False

        self.model_dir = model_dir
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.log_dir = os.path.join(self.model_dir, "logs")
        self.video_dir = os.path.join(self.model_dir, "videos")
        self.dirs = [self.checkpoint_dir, self.log_dir, self.video_dir]
        for directory in self.dirs:
            os.makedirs(directory, exist_ok=True)

    def init_session(self, sess=None, init_logging=True):
        if sess is not None:
            raise ValueError("PyTorch PPO does not support TensorFlow sessions.")
        self._logging_enabled = init_logging
        if init_logging and SummaryWriter is not None:
            self.train_writer = SummaryWriter(self.log_dir)
        else:
            self.train_writer = None

    def _checkpoint_path(self, step):
        return os.path.join(self.checkpoint_dir, f"model-{step}.pt")

    def _checkpoint_step(self, checkpoint_path):
        filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return int(filename.rsplit("-", 1)[-1])

    def _set_learning_rate(self):
        lr = self.initial_learning_rate * (self.lr_decay ** self.episode_idx)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _to_tensor(self, array):
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _clip_actions(self, actions):
        low = self.policy.action_low
        high = self.policy.action_high
        return torch.max(torch.min(actions, high), low)

    def _distribution_log_prob(self, distribution, actions):
        return distribution.log_prob(actions).sum(dim=-1)

    def _record_metric(self, name, value):
        self._episodic_metric_totals[name] = self._episodic_metric_totals.get(name, 0.0) + float(value)
        self._episodic_metric_counts[name] = self._episodic_metric_counts.get(name, 0) + 1

    def _write_histogram(self, name, values, step):
        if self.train_writer is None:
            return
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.train_writer.add_histogram(name, values, step)

    def save(self):
        checkpoint_path = self._checkpoint_path(self.episode_idx)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_old_state_dict": self.policy_old.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_step_idx": self.train_step_idx,
                "predict_step_idx": self.predict_step_idx,
                "episode_idx": self.episode_idx,
            },
            checkpoint_path,
        )
        print(f"Model checkpoint saved to {checkpoint_path}")

    def load_latest_checkpoint(self):
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model-*.pt"))
        if not checkpoint_files:
            return False

        checkpoint_path = max(checkpoint_files, key=self._checkpoint_step)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(
            checkpoint.get("policy_old_state_dict", checkpoint["policy_state_dict"])
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_idx = int(checkpoint.get("train_step_idx", 0))
        self.predict_step_idx = int(checkpoint.get("predict_step_idx", 0))
        self.episode_idx = int(checkpoint.get("episode_idx", 0))
        print(f"Model checkpoint restored from {checkpoint_path}")
        return True

    def train(self, input_states, taken_actions, returns, advantage):
        input_states = self._to_tensor(input_states)
        taken_actions = self._to_tensor(taken_actions)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor(advantage, dtype=torch.float32, device=self.device)

        self.policy.train()
        self._set_learning_rate()

        distribution, values, action_mean = self.policy(input_states)
        with torch.no_grad():
            old_distribution, _, _ = self.policy_old(input_states)

        log_prob = self._distribution_log_prob(distribution, taken_actions)
        old_log_prob = self._distribution_log_prob(old_distribution, taken_actions)
        prob_ratio = torch.exp(log_prob - old_log_prob)

        clipped_ratio = torch.clamp(prob_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        policy_loss = torch.min(prob_ratio * advantage, clipped_ratio * advantage).mean()
        value_loss = torch.mean((values - returns) ** 2) * self.value_scale
        entropy_loss = distribution.entropy().sum(dim=-1).mean() * self.entropy_scale
        loss = -policy_loss + value_loss - entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._record_metric("train_loss/policy", policy_loss.item())
        self._record_metric("train_loss/value", value_loss.item())
        self._record_metric("train_loss/entropy", entropy_loss.item())
        self._record_metric("train_loss/loss", loss.item())
        self._record_metric("train/prob_ratio", prob_ratio.mean().item())
        self._record_metric("train/returns", returns.mean().item())
        self._record_metric("train/advantage", advantage.mean().item())
        self._record_metric("train/learning_rate", self.optimizer.param_groups[0]["lr"])

        for action_idx in range(self.action_space.shape[0]):
            self._record_metric(
                f"train_actor/action_{action_idx}/taken_actions",
                taken_actions[:, action_idx].mean().item(),
            )
            self._record_metric(
                f"train_actor/action_{action_idx}/mean",
                action_mean[:, action_idx].mean().item(),
            )
            self._record_metric(
                f"train_actor/action_{action_idx}/std",
                torch.exp(self.policy.action_logstd[action_idx]).item(),
            )

        if self.train_writer is not None:
            step = self.train_step_idx
            for action_idx in range(self.action_space.shape[0]):
                self._write_histogram(
                    f"train_actor_step/action_{action_idx}/taken_actions",
                    taken_actions[:, action_idx],
                    step,
                )
                self._write_histogram(
                    f"train_actor_step/action_{action_idx}/mean",
                    action_mean[:, action_idx],
                    step,
                )
                self._write_histogram(
                    f"train_actor_step/action_{action_idx}/std",
                    torch.exp(self.policy.action_logstd[action_idx]).detach().cpu().view(1),
                    step,
                )
            self._write_histogram("train_step/input_states", input_states, step)
            self._write_histogram("train_step/prob_ratio", prob_ratio, step)

        self.train_step_idx += 1

    def predict(self, input_states, greedy=False, write_to_summary=False):
        inputs = self._to_tensor(input_states)
        self.policy.eval()
        with torch.no_grad():
            distribution, value, action_mean = self.policy(inputs)
            raw_action = action_mean if greedy else distribution.sample()
            action = self._clip_actions(raw_action)

        if write_to_summary and self.train_writer is not None:
            step = self.predict_step_idx
            for action_idx in range(self.action_space.shape[0]):
                self.train_writer.add_scalar(
                    f"predict_actor/action_{action_idx}/sampled_action",
                    float(action[0, action_idx].item()),
                    step,
                )
                self.train_writer.add_scalar(
                    f"predict_actor/action_{action_idx}/mean",
                    float(action_mean[0, action_idx].item()),
                    step,
                )
                self.train_writer.add_scalar(
                    f"predict_actor/action_{action_idx}/std",
                    float(torch.exp(self.policy.action_logstd[action_idx]).item()),
                    step,
                )
            self.predict_step_idx += 1

        action_np = action.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        if action_np.shape[0] == 1:
            return action_np[0], value_np[0]
        return action_np, value_np

    def get_episode_idx(self):
        return self.episode_idx

    def get_train_step_idx(self):
        return self.train_step_idx

    def get_predict_step_idx(self):
        return self.predict_step_idx

    def write_value_to_summary(self, summary_name, value, step):
        if self.train_writer is not None:
            self.train_writer.add_scalar(summary_name, value, step)

    def write_dict_to_summary(self, summary_name, params, step):
        if self.train_writer is not None:
            lines = [f"{key}: {value}" for key, value in params.items()]
            self.train_writer.add_text(summary_name, "\n".join(lines), step)

    def write_episodic_summaries(self):
        if self.train_writer is not None:
            for name, total in self._episodic_metric_totals.items():
                count = max(self._episodic_metric_counts.get(name, 1), 1)
                self.train_writer.add_scalar(name, total / count, self.episode_idx)
            self.train_writer.flush()
        self._episodic_metric_totals = {}
        self._episodic_metric_counts = {}
        self.episode_idx += 1

    def update_old_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
