"""
WGAN-GP Face Generation — Model Architecture
=============================================
Wasserstein GAN with Gradient Penalty for high-quality 256x256 face synthesis.

Key design choices:
  - Progressive growing of generator/discriminator (128 -> 256)
  - Spectral normalisation in discriminator for training stability
  - Gradient penalty enforces Lipschitz constraint (replaces clipping)
  - Minibatch standard deviation in discriminator to prevent mode collapse
  - Equalized learning rate (He init scaled at runtime)

Reference: Gulrajani et al. (2017) "Improved Training of Wasserstein GANs"
           Karras et al. (2018) "Progressive Growing of GANs"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

class PixelNorm(nn.Module):
    """Normalise activation vector to unit length per pixel."""
    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8).sqrt()


class MinibatchStdDev(nn.Module):
    """Append per-group feature std to feature maps — discourages mode collapse."""
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        g = min(self.group_size, N)
        y = x.view(g, -1, C, H, W).float()
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0)
        y = (y + 1e-8).sqrt().mean([1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, H, W)
        return torch.cat([x, y.to(x.dtype)], dim=1)


class ScaledConv2d(nn.Module):
    """Conv2d with equalized learning rate (He init, scale at forward time)."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, bias=True):
        super().__init__()
        self.scale = np.sqrt(2.0 / (in_ch * kernel * kernel))
        self.conv  = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=bias)
        nn.init.normal_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x * self.scale)


# ── Generator ─────────────────────────────────────────────────────────────────

class GeneratorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else nn.Identity()
        self.conv1 = ScaledConv2d(in_ch, out_ch)
        self.conv2 = ScaledConv2d(out_ch, out_ch)
        self.norm  = PixelNorm()

    def forward(self, x):
        x = self.up(x)
        x = F.leaky_relu(self.norm(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.norm(self.conv2(x)), 0.2)
        return x


class Generator(nn.Module):
    """
    Maps a latent vector z (512-d) to a 256x256 RGB face.
    Architecture: learned const -> 4x4 -> 8x8 -> ... -> 256x256
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        # 4x4 learned constant
        self.const = nn.Parameter(torch.ones(1, 512, 4, 4))
        self.latent_proj = nn.Linear(latent_dim, 512)

        self.blocks = nn.ModuleList([
            GeneratorBlock(512, 512, upsample=False),   # 4x4
            GeneratorBlock(512, 512),                    # 8x8
            GeneratorBlock(512, 256),                    # 16x16
            GeneratorBlock(256, 128),                    # 32x32
            GeneratorBlock(128, 64),                     # 64x64
            GeneratorBlock(64,  32),                     # 128x128
            GeneratorBlock(32,  16),                     # 256x256
        ])
        self.to_rgb = ScaledConv2d(16, 3, kernel=1, pad=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        style = self.latent_proj(z).unsqueeze(-1).unsqueeze(-1)
        x = self.const.expand(z.size(0), -1, -1, -1) + style
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.to_rgb(x))


# ── Discriminator ─────────────────────────────────────────────────────────────

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.down  = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.skip  = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0)) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        skip = self.down(self.skip(x))
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.down(F.leaky_relu(self.conv2(x), 0.2))
        return (x + skip) * (1 / np.sqrt(2))


class Discriminator(nn.Module):
    """
    Maps a 256x256 image to a scalar Wasserstein score.
    Uses spectral norm + minibatch std dev for stability.
    No sigmoid — raw score for Wasserstein loss.
    """
    def __init__(self):
        super().__init__()
        self.from_rgb = nn.utils.spectral_norm(nn.Conv2d(3, 16, 1))
        self.blocks   = nn.ModuleList([
            DiscriminatorBlock(16,  32),   # 256 -> 128
            DiscriminatorBlock(32,  64),   # 128 -> 64
            DiscriminatorBlock(64,  128),  # 64  -> 32
            DiscriminatorBlock(128, 256),  # 32  -> 16
            DiscriminatorBlock(256, 512),  # 16  -> 8
            DiscriminatorBlock(512, 512),  # 8   -> 4
        ])
        self.mbstd   = MinibatchStdDev(group_size=4)
        self.final   = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(513, 512, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.from_rgb(x), 0.2)
        for block in self.blocks:
            x = block(x)
        x = self.mbstd(x)
        return self.final(x)


# ── Gradient Penalty ──────────────────────────────────────────────────────────

def gradient_penalty(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor, device: str) -> torch.Tensor:
    """
    WGAN-GP gradient penalty: enforce ||grad D(x_hat)||_2 = 1.
    x_hat = epsilon * real + (1-epsilon) * fake   (random interpolation)
    """
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_hat = disc(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


if __name__ == "__main__":
    from model import Generator, Discriminator, gradient_penalty
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(512).to(device)
    D = Discriminator().to(device)
    z = torch.randn(4, 512, device=device)
    fake = G(z)
    score = D(fake)
    print(f"Generator output : {fake.shape}")
    print(f"Discriminator score: {score.shape}")
    G_params = sum(p.numel() for p in G.parameters()) / 1e6
    D_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"Generator params : {G_params:.2f}M")
    print(f"Discriminator params: {D_params:.2f}M")
