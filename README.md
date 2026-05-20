# Deepfake WGAN-GP Face Generation

Wasserstein GAN with Gradient Penalty for photorealistic 256x256 face synthesis. Part of MPhil research on hybrid multi-modal deepfake detection.

## Key Features
- Progressive-growing-inspired architecture
- Spectral normalisation + MinibatchStdDev for stability
- Equalized learning rate (ProGAN technique)
- EMA of generator weights for smooth sampling
- FID/IS evaluation metrics

## Architecture
- Generator: const(4x4) -> progressive upsample -> 256x256 (512-d latent)
- Discriminator: residual downsampling + spectral norm + PatchGAN output
- Loss: Wasserstein distance + GP(lambda=10)

## Training
```bash
python train.py --dataset celeba --data ./data --epochs 100 --batch_size 16
python train.py --dataset ffhq   --data ./data/ffhq --epochs 100 --batch_size 8
```

## Generation
```bash
# Random faces
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --n 16 --out faces.png
# Latent interpolation
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --n 10 --interpolate --out interp.png
```

## GAN Artifacts for Detection Research
- Checkerboard artifacts from transposed convolutions
- Spectral inconsistencies in high-frequency bands
- Eye symmetry violations
- Unnatural skin texture periodicity

---
MPhil Research | [Dnshitobu](https://github.com/Dnshitobu)
