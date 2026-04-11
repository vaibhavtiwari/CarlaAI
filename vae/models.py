import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def _as_shape(shape):
    return tuple(int(x) for x in shape)


def _to_tensor_nhwc(array, device):
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _nhwc_to_nchw(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()


def _nchw_to_nhwc(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()


def kl_divergence(mean, logstd_sq):
    return -0.5 * torch.sum(1.0 + logstd_sq - mean.pow(2) - logstd_sq.exp(), dim=1)


def bce_loss(labels, logits, _targets):
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none")


def bce_loss_v2(labels, _logits, targets, epsilon=1e-10):
    return -(labels * torch.log(epsilon + targets) + (1 - labels) * torch.log(epsilon + 1 - targets))


def mse_loss(labels, _logits, targets):
    return (labels - targets).pow(2)


class VAE(nn.Module):
    """
    Base variational autoencoder class with a NumPy-facing API matching the old
    TensorFlow implementation closely enough for the training and CARLA helpers.
    """

    def __init__(
        self,
        source_shape,
        target_shape,
        encoder,
        decoder,
        encoded_size,
        z_dim=512,
        beta=1.0,
        learning_rate=1e-4,
        lr_decay=0.98,
        kl_tolerance=0.0,
        model_dir=".",
        loss_fn=bce_loss,
        training=True,
        device=None,
        **_,
    ):
        super().__init__()
        self.source_shape = _as_shape(source_shape)
        self.target_shape = _as_shape(target_shape)
        self.z_dim = int(z_dim)
        self.beta = beta
        self.kl_tolerance = kl_tolerance
        self.model_dir = model_dir
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.log_dir = os.path.join(self.model_dir, "logs")
        self.dirs = [self.checkpoint_dir, self.log_dir]
        for directory in self.dirs:
            os.makedirs(directory, exist_ok=True)

        self.encoder = encoder
        self.decoder = decoder
        self.mean = nn.Linear(encoded_size, self.z_dim)
        self.logstd_sq = nn.Linear(encoded_size, self.z_dim)
        self.loss_fn = loss_fn
        self.step_idx = 0
        self.sample = np.zeros((1, self.z_dim), dtype=np.float32)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

        self.optimizer = Adam(self.parameters(), lr=learning_rate) if training else None
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_decay) if training else None
        self.train_writer = None
        self.val_writer = None

    def init_session(self, sess=None, init_logging=True):
        if sess is not None:
            raise ValueError("PyTorch VAE does not support TensorFlow sessions.")
        if init_logging and SummaryWriter is None:
            print(
                "TensorBoard logging is disabled because torch.utils.tensorboard.SummaryWriter "
                "is unavailable in this environment.",
                flush=True,
            )
            return
        if init_logging and SummaryWriter is not None:
            self.train_writer = SummaryWriter(os.path.join(self.log_dir, "train"))
            self.val_writer = SummaryWriter(os.path.join(self.log_dir, "val"))

    def _prepare_source(self, source_states):
        source = _to_tensor_nhwc(source_states, self.device)
        self._verify_range(source)
        return source

    def _prepare_target(self, target_states):
        target = _to_tensor_nhwc(target_states, self.device)
        self._verify_range(target)
        return target

    def _verify_range(self, tensor):
        if torch.any(tensor < 0) or torch.any(tensor > 1):
            raise ValueError(
                "VAE inputs must be in [0, 1]. "
                f"Got min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
            )

    def encode_tensor(self, source_states):
        encoded = self.encoder(_nhwc_to_nchw(source_states))
        return self.mean(encoded), self.logstd_sq(encoded)

    def reparameterize(self, mean, logstd_sq):
        if self.training:
            std = torch.exp(0.5 * logstd_sq)
            return mean + torch.randn_like(std) * std
        return mean

    def decode_tensor(self, z):
        logits = self.decoder(z)
        targets = torch.sigmoid(logits)
        return logits, targets

    def forward(self, source_states):
        mean, logstd_sq = self.encode_tensor(source_states)
        z = self.reparameterize(mean, logstd_sq)
        logits, targets = self.decode_tensor(z)
        return logits, targets, mean, logstd_sq

    def _losses(self, source_states, target_states):
        source = self._prepare_source(source_states)
        target = self._prepare_target(target_states)

        logits_nchw, reconstructed_nchw, mean, logstd_sq = self(source)
        logits = _nchw_to_nhwc(logits_nchw).reshape(source.shape[0], -1)
        reconstructed = _nchw_to_nhwc(reconstructed_nchw).reshape(source.shape[0], -1)
        flattened_target = target.reshape(target.shape[0], -1)

        reconstruction_loss = self.loss_fn(flattened_target, logits, reconstructed).sum(dim=1).mean()
        kl_loss = kl_divergence(mean, logstd_sq)
        if self.kl_tolerance > 0:
            kl_loss = torch.maximum(
                kl_loss,
                torch.full_like(kl_loss, self.kl_tolerance * self.z_dim),
            )
        kl_loss = kl_loss.mean()
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def _iterate_minibatches(self, source, target=None, batch_size=None, shuffle=True):
        if target is None:
            for source_batch, target_batch in source:
                yield source_batch, target_batch
            return

        if batch_size is None:
            raise ValueError("batch_size is required when training from arrays.")
        indices = np.arange(len(source))
        if shuffle:
            np.random.shuffle(indices)
        for begin in range(0, len(source), batch_size):
            mb_idx = indices[begin:begin + batch_size]
            if len(mb_idx) > 0:
                yield source[mb_idx], target[mb_idx]

    def _write_scalars(self, writer, reconstruction_loss, kl_loss):
        if writer is None:
            return
        writer.add_scalar("reconstruction_loss", reconstruction_loss, self.step_idx)
        writer.add_scalar("kl_loss", kl_loss, self.step_idx)
        if self.optimizer is not None:
            writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.step_idx)
        writer.flush()

    def _run_epoch(self, source, target=None, batch_size=None, progress=False, training=False):
        if training:
            self.train()
        else:
            self.eval()

        reconstruction_losses = []
        kl_losses = []
        num_batches = len(source) if hasattr(source, "__len__") and target is None else None
        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_idx, (source_batch, target_batch) in enumerate(
                self._iterate_minibatches(source, target, batch_size, shuffle=training),
                start=1,
            ):
                if training:
                    self.optimizer.zero_grad()

                loss, reconstruction_loss, kl_loss = self._losses(source_batch, target_batch)

                if training:
                    loss.backward()
                    self.optimizer.step()

                reconstruction_losses.append(reconstruction_loss.item())
                kl_losses.append(kl_loss.item())

                if progress and (batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == num_batches):
                    suffix = f"/{num_batches}" if num_batches is not None else ""
                    split = "train" if training else "val"
                    print(
                        f"  {split} batch {batch_idx}{suffix}: "
                        f"reconstruction={reconstruction_loss.item():.4f}, "
                        f"kl={kl_loss.item():.4f}",
                        flush=True,
                    )

        if not reconstruction_losses:
            split = "training" if training else "validation"
            raise ValueError(f"No {split} batches were created. Check the dataset size and batch_size.")

        return float(np.mean(reconstruction_losses)), float(np.mean(kl_losses))

    def save(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model-{self.step_idx}.pt")
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "step_idx": self.step_idx,
                "source_shape": self.source_shape,
                "target_shape": self.target_shape,
                "z_dim": self.z_dim,
                "beta": self.beta,
                "kl_tolerance": self.kl_tolerance,
            },
            checkpoint_path,
        )
        print("Model checkpoint saved to {}".format(checkpoint_path))

    def load_latest_checkpoint(self):
        checkpoint_paths = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not checkpoint_paths:
            return False
        checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        if self.optimizer is not None and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step_idx = int(checkpoint.get("step_idx", 0))
        print("Model checkpoint restored from {}".format(checkpoint_path))
        return True

    def generate_from_latent(self, z):
        self.eval()
        with torch.no_grad():
            z = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            _, reconstructed = self.decode_tensor(z)
            return _nchw_to_nhwc(reconstructed).cpu().numpy()

    def sample_latent(self, source_states):
        self.eval()
        with torch.no_grad():
            source = self._prepare_source(source_states)
            mean, logstd_sq = self.encode_tensor(source)
            std = torch.exp(0.5 * logstd_sq)
            z = mean + torch.randn_like(std) * std
            return z.cpu().numpy()

    def reconstruct(self, source_states):
        self.eval()
        with torch.no_grad():
            source = self._prepare_source(source_states)
            _, reconstructed, _, _ = self(source)
            return _nchw_to_nhwc(reconstructed).cpu().numpy()

    def encode(self, source_states):
        self.eval()
        with torch.no_grad():
            source = self._prepare_source(source_states)
            mean, _ = self.encode_tensor(source)
            return mean.cpu().numpy()

    def get_step_idx(self):
        return self.step_idx

    def train_one_epoch(self, train_source, train_target=None, batch_size=None, progress=False):
        mean_reconstruction_loss, mean_kl_loss = self._run_epoch(
            train_source,
            train_target,
            batch_size=batch_size,
            progress=progress,
            training=True,
        )
        if self.scheduler is not None:
            self.scheduler.step()
        self._write_scalars(self.train_writer, mean_reconstruction_loss, mean_kl_loss)
        self.step_idx += 1

    def evaluate(self, val_source, val_target=None, batch_size=None, progress=False):
        mean_reconstruction_loss, mean_kl_loss = self._run_epoch(
            val_source,
            val_target,
            batch_size=batch_size,
            progress=progress,
            training=False,
        )
        self._write_scalars(self.val_writer, mean_reconstruction_loss, mean_kl_loss)
        return mean_reconstruction_loss, mean_kl_loss


class ConvEncoder(nn.Module):
    def __init__(self, source_shape):
        super().__init__()
        channels = source_shape[-1]
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, source_shape[-1], source_shape[0], source_shape[1])
            encoded = self.net(dummy)
        self.encoded_shape = tuple(encoded.shape[1:])
        self.encoded_size = int(np.prod(self.encoded_shape))

    def forward(self, x):
        return torch.flatten(self.net(x), start_dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, z_dim, encoded_shape, target_shape):
        super().__init__()
        self.encoded_shape = _as_shape(encoded_shape)
        self.dense = nn.Linear(z_dim, int(np.prod(self.encoded_shape)))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, target_shape[-1], kernel_size=4, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *self.encoded_shape)
            output_shape = _nchw_to_nhwc(self.net(dummy)).shape[1:]
        if tuple(output_shape) != tuple(target_shape):
            raise ValueError(f"ConvVAE decoder output shape {tuple(output_shape)} != target shape {tuple(target_shape)}")

    def forward(self, z):
        x = self.dense(z).reshape(z.shape[0], *self.encoded_shape)
        return self.net(x)


class ConvVAE(VAE):
    """
    Convolutional VAE architecture from the original TensorFlow version.
    This is tuned for 80x160 source images.
    """

    def __init__(self, source_shape, target_shape=None, **kwargs):
        source_shape = _as_shape(source_shape)
        target_shape = source_shape if target_shape is None else _as_shape(target_shape)
        z_dim = int(kwargs.get("z_dim", 512))
        encoder = ConvEncoder(source_shape)
        decoder = ConvDecoder(z_dim, encoder.encoded_shape, target_shape)
        super().__init__(
            source_shape,
            target_shape,
            encoder,
            decoder,
            encoder.encoded_size,
            **kwargs,
        )


class MlpEncoder(nn.Module):
    def __init__(self, source_shape, encoder_sizes):
        super().__init__()
        layers = []
        input_size = int(np.prod(source_shape))
        for hidden_size in encoder_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.net = nn.Sequential(*layers)
        self.encoded_size = encoder_sizes[-1]

    def forward(self, x):
        x = _nchw_to_nhwc(x).reshape(x.shape[0], -1)
        return self.net(x)


class MlpDecoder(nn.Module):
    def __init__(self, target_shape, decoder_sizes):
        super().__init__()
        self.target_shape = _as_shape(target_shape)
        layers = []
        input_size = decoder_sizes[0]
        for hidden_size in decoder_sizes[1:]:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, int(np.prod(self.target_shape))))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        x = x.reshape(z.shape[0], *self.target_shape)
        return _nhwc_to_nchw(x)


class MlpVAE(VAE):
    """
    Multi-layered perceptron VAE.
    """

    def __init__(
        self,
        source_shape,
        target_shape=None,
        encoder_sizes=(512, 256),
        decoder_sizes=(256, 512),
        **kwargs,
    ):
        source_shape = _as_shape(source_shape)
        target_shape = source_shape if target_shape is None else _as_shape(target_shape)
        encoder = MlpEncoder(source_shape, encoder_sizes)
        decoder = MlpDecoder(target_shape, (kwargs.get("z_dim", 512), *decoder_sizes))
        super().__init__(
            source_shape,
            target_shape,
            encoder,
            decoder,
            encoder.encoded_size,
            **kwargs,
        )
