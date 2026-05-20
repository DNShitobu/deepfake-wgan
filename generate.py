"""WGAN-GP Face Generation — Inference Script"""
import argparse, torch
from pathlib import Path
from torchvision.utils import save_image
from model import Generator

def load_generator(ckpt, latent_dim=512, use_ema=True, device="cpu"):
    G = Generator(latent_dim).to(device)
    ckpt = torch.load(ckpt, map_location=device)
    weights = ckpt.get("G_ema" if use_ema else "G", ckpt.get("G"))
    G.load_state_dict(weights)
    G.eval()
    return G

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=16, help="Number of faces to generate")
    p.add_argument("--out", default="generated_faces.png")
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--interpolate", action="store_true", help="Generate latent space interpolation")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    G = load_generator(args.ckpt, args.latent_dim, device=device)
    with torch.no_grad():
        if args.interpolate:
            z1 = torch.randn(1, args.latent_dim, device=device)
            z2 = torch.randn(1, args.latent_dim, device=device)
            alphas = torch.linspace(0, 1, args.n, device=device)
            z = torch.stack([z1 * (1-a) + z2 * a for a in alphas]).squeeze(1)
        else:
            z = torch.randn(args.n, args.latent_dim, device=device)
        imgs = G(z) * 0.5 + 0.5
    nrow = 8 if args.n >= 8 else args.n
    save_image(imgs, args.out, nrow=nrow)
    print(f"Saved {args.n} faces to {args.out}")
