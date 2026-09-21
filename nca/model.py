"""Audio-conditioned Neural Cellular Automaton.

The core idea: every cell runs the SAME tiny neural network on its local
neighbourhood. Global form (a diatom frustule) emerges from iterated local
rules — morphogenesis, not blueprinting.

"Feed audio into the model's weights" is implemented as FiLM conditioning:
an audio feature vector is mapped to per-neuron gain (gamma) and bias (beta)
that modulate the update MLP's hidden layer. Loud/bright/rhythmic sound
literally retunes the local rule each step, so morphology bends with sound
while the rule stays local and shared.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Fixed perception kernels: identity, Sobel X, Sobel Y.
IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
SOBEL_Y = SOBEL_X.T.clone()

# Channel semantics (convention used by visualisation + damage + audio):
#   0:2  RGB  — visible pigment (fucoxanthin golds, glass blues)
#   3    A    — alpha / maturity. alive = maxpool(A) > 0.1
#   4:7   DNA — genome channels. Constant per organism, steer pattern fate.
#   8:10  MORPH — slow morphogen gradients (auxiliary memory)
#   11   AUDIO — audio-energy sense channel (written from conditioning vector)
#   12:15 HIDDEN — free latent / silica-deposition state
N_CHANNELS = 16
N_AUDIO_DIMS = 8
RGB = (0, 1, 2)
ALPHA = 3
DNA = (4, 5, 6, 7)
AUDIO_CH = 11


class DiatomNCA(nn.Module):
    """Single shared local rule with audio FiLM conditioning."""

    def __init__(
        self,
        num_channels: int = N_CHANNELS,
        hidden_dim: int = 128,
        audio_dim: int = N_AUDIO_DIMS,
        fire_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.fire_rate = fire_rate

        # Perception is fixed (not learned): each channel sees itself +
        # x/y gradients of its 3x3 neighbourhood.
        kernel = torch.stack([IDENTITY, SOBEL_X, SOBEL_Y])  # (3,3,3)
        kernel = kernel[:, None, :, :].repeat(num_channels, 1, 1, 1)  # (C*3,1,3,3)
        self.register_buffer("perception_kernel", kernel)

        self.fc1 = nn.Linear(num_channels * 3, hidden_dim)
        self.relu = nn.ReLU()
        # Audio -> FiLM params (gamma, beta) for the hidden layer.
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, num_channels)

        # Small (not zero) init: the untrained rule already grows a little
        # "primordial soup", and audio conditioning bends growth from step
        # one. (Pure zero-init would make silence and sound identical until
        # training; we prefer an alive-at-birth creature.)
        nn.init.normal_(self.fc2.weight, std=0.05)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.audio_encoder.weight, std=0.2)
        nn.init.zeros_(self.audio_encoder.bias)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise 3x3 perception. x: (B,C,H,W) -> (B,C*3,H,W)."""
        perceps = F.conv2d(x, self.perception_kernel, padding=1, groups=self.num_channels)
        return perceps.permute(0, 2, 3, 1)  # (B,H,W,C*3)

    def update(
        self, x: torch.Tensor, audio: torch.Tensor | None = None, fire_rate: float | None = None
    ) -> torch.Tensor:
        """One CA step. audio: (B,audio_dim) or None for silence."""
        b, c, h, w = x.shape
        rate = self.fire_rate if fire_rate is None else fire_rate
        device, dtype = x.device, x.dtype

        p = self.perceive(x)  # (B,H,W,C*3)
        hidden = self.relu(self.fc1(p))  # (B,H,W,hidden)

        if audio is None:
            audio = torch.zeros(b, self.audio_dim, device=device, dtype=dtype)
        film = self.audio_encoder(audio)  # (B, hidden*2)
        gamma, beta = film.chunk(2, dim=-1)  # each (B,hidden)
        gamma = gamma[:, None, None, :]  # broadcast over H,W
        beta = beta[:, None, None, :]
        hidden = hidden * (1.0 + gamma) + beta

        dx = self.fc2(hidden).permute(0, 3, 1, 2)  # (B,C,H,W)

        # Stochastic update: each cell fires with probability `rate`.
        stochastic = (torch.rand(b, 1, h, w, device=device) < rate).to(dtype)
        # Alive mask: a cell survives iff some neighbour has alpha > 0.1.
        alive = (F.max_pool2d(x[:, ALPHA : ALPHA + 1], 3, stride=1, padding=1) > 0.1).to(dtype)
        return x + dx * stochastic * alive

    def forward(
        self, x: torch.Tensor, audio: torch.Tensor | None = None, steps: int = 1, **kwargs
    ) -> torch.Tensor:
        for _ in range(steps):
            x = self.update(x, audio=audio, **kwargs)
        return x

    # -- helpers ---------------------------------------------------------
    def seed(
        self,
        batch: int,
        size: int,
        device: torch.device | str = "cpu",
        genome: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single living cell in the centre. Optional fixed DNA genome (B,4)."""
        x = torch.zeros(batch, self.num_channels, size, size, device=device)
        x[:, ALPHA, size // 2, size // 2] = 1.0
        if genome is None:
            genome = torch.randn(batch, len(DNA), device=device) * 0.2
        for i, ch in enumerate(DNA):
            x[:, ch, size // 2, size // 2] = genome[:, i]
        return x

    @torch.no_grad()
    def grow(
        self,
        steps: int,
        size: int = 48,
        batch: int = 1,
        audio: torch.Tensor | None = None,
        audio_schedule: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
        genome: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Roll out growth; returns full trajectory (B,T,C,H,W) incl. t=0."""
        self.eval()
        x = self.seed(batch, size, device=device, genome=genome)
        traj = [x.clone()]
        for t in range(steps):
            a = audio_schedule[t] if audio_schedule is not None else audio
            if a is not None and a.dim() == 2:
                pass  # already (B,audio_dim)
            x = self.update(x, audio=a)
            traj.append(x.clone())
        return torch.stack(traj, dim=1)
