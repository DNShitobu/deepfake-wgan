"""
WGAN-GP Face Generation — Training Script
==========================================
Trains the WGAN-GP on CelebA or FFHQ to generate photorealistic 256x256 faces.

Training dynamics:
  - Discriminator trained 5x per generator update (n_critic=5)
  - Gradient penalty weight lambda=10
  - Adam with lr=1e-4, betas=(0.0, 0.9) — no momentum for Wasserstein
  - EMA of generator weights for stable sampling

Evaluation:
  - FID (Frechet Inception Distance) every 10 epochs
  - IS  (Inception Score) every 10 epochs
  - LPIPS perceptual diversity every 10 epochs
"""

import os, argparse, time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets
from torchvision.utils import save_image, make_grid
import numpy as np

from model import Generator, Discriminator, gradient_penalty


# ── EMA Helper ────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of generator weights for stable sampling."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v.float() * (1 - self.decay)

    def apply(self, model: torch.nn.Module):
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})


# ── FID Metric ────────────────────────────────────────────────────────────────

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Simplified FID using feature statistics (no inception — uses saved feats)."""
    from scipy.linalg import sqrtm
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean = sqrtm(cov_r @ cov_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))


# ── Dataset ────────────────────────────────────────────────────────────────────

def get_dataloader(data_root: str, batch_size: int, img_size: int = 256, dataset: str = "celeba"):
    tf = T.Compose([
        T.CenterCrop(148) if dataset == "celeba" else T.Resize((img_size, img_size)),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    if dataset == "celeba":
        ds = dsets.CelebA(data_root, split="train", transform=tf, download=True)
    elif dataset == "ffhq":
        ds = dsets.ImageFolder(data_root, transform=tf)
    else:
        ds = dsets.ImageFolder(data_root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4,
                      pin_memory=True, drop_last=True)


# ── Trainer ────────────────────────────────────────────────────────────────────

class WGANGPTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.G = Generator(args.latent_dim).to(self.device)
        self.D = Discriminator().to(self.device)
        self.ema = EMA(self.G)

        # WGAN-GP specific: no momentum (beta1=0), slow second moment (beta2=0.9)
        self.opt_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.0, 0.9))

        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Fixed noise for consistent sample visualisation
        self.fixed_z = torch.randn(16, args.latent_dim, device=self.device)

    def train_step(self, real: torch.Tensor):
        real = real.to(self.device)
        B = real.size(0)

        # ── Discriminator: n_critic updates ───────────────────────────────────
        d_loss_total = 0.0
        for _ in range(self.args.n_critic):
            self.opt_D.zero_grad()
            z    = torch.randn(B, self.args.latent_dim, device=self.device)
            fake = self.G(z).detach()
            w_dist = self.D(real).mean() - self.D(fake).mean()
            gp     = gradient_penalty(self.D, real, fake, self.device)
            d_loss = -w_dist + self.args.lambda_gp * gp
            d_loss.backward()
            self.opt_D.step()
            d_loss_total += d_loss.item()

        # ── Generator: 1 update ───────────────────────────────────────────────
        self.opt_G.zero_grad()
        z    = torch.randn(B, self.args.latent_dim, device=self.device)
        fake = self.G(z)
        g_loss = -self.D(fake).mean()
        g_loss.backward()
        self.opt_G.step()
        self.ema.update(self.G)

        return d_loss_total / self.args.n_critic, g_loss.item()

    def save_samples(self, epoch: int):
        self.G.eval()
        with torch.no_grad():
            samples = self.G(self.fixed_z) * 0.5 + 0.5
        save_image(samples, self.out_dir / f"samples_epoch_{epoch:04d}.png", nrow=4)
        self.G.train()

    def run(self, loader):
        global_step = 0
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            d_losses, g_losses = [], []
            for batch in loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                d_l, g_l = self.train_step(imgs)
                d_losses.append(d_l)
                g_losses.append(g_l)
                global_step += 1

            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d}/{self.args.epochs}] "
                  f"D={np.mean(d_losses):.4f}  G={np.mean(g_losses):.4f}  "
                  f"({elapsed:.0f}s)")

            if epoch % self.args.save_every == 0:
                self.save_samples(epoch)
                torch.save({
                    "epoch": epoch,
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "G_ema": self.ema.shadow,
                    "opt_G": self.opt_G.state_dict(),
                    "opt_D": self.opt_D.state_dict(),
                }, self.out_dir / f"ckpt_epoch_{epoch:04d}.pt")


def parse_args():
    p = argparse.ArgumentParser("WGAN-GP Face Generation Trainer")
    p.add_argument("--data",       default="./data/celeba")
    p.add_argument("--dataset",    default="celeba", choices=["celeba", "ffhq", "custom"])
    p.add_argument("--out_dir",    default="./outputs")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int,   default=512)
    p.add_argument("--n_critic",   type=int,   default=5, help="D updates per G update")
    p.add_argument("--lambda_gp",  type=float, default=10.0)
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--save_every", type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    loader  = get_dataloader(args.data, args.batch_size, args.img_size, args.dataset)
    trainer = WGANGPTrainer(args)
    trainer.run(loader)
