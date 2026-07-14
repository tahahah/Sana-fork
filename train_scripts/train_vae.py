"""Train the TAESD VAE on Pacman dataset frames.

Fine-tunes the pretrained madebyollin/taesd AutoencoderTiny on Pacman game frames
using reconstruction loss + KL regularization. This should produce cleaner latents
than the generic pretrained VAE, reducing noise in the diffusion model output.

Usage:
    PYTHONPATH=. python3 train_scripts/train_vae.py \
        --resolution 256 \
        --batch_size 64 \
        --num_epochs 50 \
        --lr 1e-4 \
        --output_dir output/vae_pacman \
        --save_every 5000
"""

import argparse
import os
import os.path as osp
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import AutoencoderTiny
from PIL import Image
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [VAE] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_square(image):
    w, h = image.size
    max_dim = max(w, h)
    padding = [(max_dim - w) // 2, (max_dim - h) // 2,
               (max_dim - w) - (max_dim - w) // 2,
               (max_dim - h) - (max_dim - h) // 2]
    return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')


def convert_to_rgb(img):
    return img.convert("RGB")


def rotate_90_clockwise(img):
    return img.rotate(90, expand=True)


class PacmanFrameDataset(IterableDataset):
    """Streams individual Pacman frames for VAE training."""

    def __init__(self, resolution=256, buffer_size=200):
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Lambda(make_square),
            transforms.Resize(resolution),
            transforms.functional.hflip,
            transforms.Lambda(rotate_90_clockwise),
            transforms.ToTensor(),
            # TAESD expects [0,1] — ToTensor() already outputs [0,1], no Normalize needed
        ])
        self.dataset = None

    def _init_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(
                "Tahahah/PacmanDataset_3",
                split="train",
                verification_mode="no_checks",
                streaming=True,
            )

    def __iter__(self):
        self._init_dataset()
        buffer = []
        for sample in self.dataset:
            try:
                frame = self.transform(sample['frame_image'])
            except Exception:
                continue
            buffer.append(frame)
            if len(buffer) >= self.buffer_size:
                batch = torch.stack(buffer)
                # Shuffle within buffer
                perm = torch.randperm(len(batch))
                for idx in perm:
                    yield batch[idx]
                buffer = []


# ── VAE wrapper with KL ──────────────────────────────────────────────────────
class TAESDTrainable(nn.Module):
    """Wraps AutoencoderTiny to add KL regularization on the latent.

    TAESD is a deterministic autoencoder (no VAE posterior). We add a soft KL
    penalty to encourage latents to be roughly N(0, 1), which matches the
    diffusion noise schedule.
    """

    def __init__(self, vae: AutoencoderTiny, kl_weight: float = 1e-4):
        super().__init__()
        self.vae = vae
        self.kl_weight = kl_weight
        # Latent statistics for KL: learnable running mean/logvar
        self.register_buffer("latent_mean", torch.zeros(1, 4, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 4, 1, 1))
        self.register_buffer("stat_momentum", torch.tensor(0.99))

    def encode(self, x):
        return self.vae.encoder(x)

    def decode(self, z):
        return self.vae.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def update_stats(self, z):
        """Update running statistics of latents for KL loss."""
        with torch.no_grad():
            batch_mean = z.mean(dim=(0, 2, 3), keepdim=True)
            batch_std = z.std(dim=(0, 2, 3), keepdim=True)
            m = self.stat_momentum
            self.latent_mean.mul_(m).add_(batch_mean * (1 - m))
            self.latent_std.mul_(m).add_(batch_std * (1 - m))

    def kl_loss(self, z):
        """KL(N(mu, sigma) || N(0, 1)) per element."""
        mu = self.latent_mean
        sigma = self.latent_std.clamp(min=1e-6)
        # KL = log(1/sigma) + (sigma^2 + mu^2 - 1) / 2
        kl = torch.log(1.0 / sigma) + (sigma.pow(2) + mu.pow(2) - 1.0) / 2.0
        return kl.mean()


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained TAESD
    logger.info(f"Loading pretrained TAESD from {args.vae_pretrained}")
    vae = AutoencoderTiny.from_pretrained(args.vae_pretrained).to(device)
    vae.train()  # Enable gradients

    # Unfreeze all VAE parameters for fine-tuning
    for param in vae.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"TAESD VAE: {total_params/1e6:.2f}M params, {trainable_params/1e6:.2f}M trainable")

    # Wrap with KL regularization
    model = TAESDTrainable(vae, kl_weight=args.kl_weight).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * 20000, eta_min=args.lr * 0.01
    )

    # Dataset
    dataset = PacmanFrameDataset(resolution=args.resolution, buffer_size=args.buffer_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")

    logger.info(f"Starting VAE training: bs={args.batch_size}, lr={args.lr}, "
                f"resolution={args.resolution}, kl_weight={args.kl_weight}")
    logger.info(f"Output dir: {args.output_dir}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for batch in dataloader:
            x = batch.to(device, non_blocking=True)

            # Forward
            with torch.amp.autocast("cuda", enabled=args.mixed_precision == "fp16"):
                recon, z = model(x)

                # Reconstruction loss: MSE in pixel space [-1, 1]
                recon_loss = F.mse_loss(recon, x)

                # KL regularization
                model.update_stats(z)
                kl_loss = model.kl_loss(z) * model.kl_weight

                loss = recon_loss + kl_loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

            if global_step % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"loss: {loss.item():.6f}, recon: {recon_loss.item():.6f}, "
                    f"kl: {kl_loss.item():.6f}, lr: {scheduler.get_last_lr()[0]:.2e}, "
                    f"latent_mean: {z.mean().item():.4f}, latent_std: {z.std().item():.4f}"
                )

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_path = osp.join(args.output_dir, f"vae_step_{global_step}.pth")
                torch.save({
                    "vae_state_dict": vae.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": global_step,
                    "loss": loss.item(),
                }, save_path)
                logger.info(f"Saved checkpoint: {save_path}")

            # Save best
            if recon_loss.item() < best_loss:
                best_loss = recon_loss.item()
                best_path = osp.join(args.output_dir, "vae_best.pth")
                torch.save({
                    "vae_state_dict": vae.state_dict(),
                    "step": global_step,
                    "loss": best_loss,
                }, best_path)

        if num_batches > 0:
            logger.info(
                f"Epoch {epoch} done | avg_loss: {epoch_loss/num_batches:.6f}, "
                f"avg_recon: {epoch_recon/num_batches:.6f}, "
                f"avg_kl: {epoch_kl/num_batches:.6f}"
            )

        # Save epoch checkpoint
        epoch_path = osp.join(args.output_dir, f"vae_epoch_{epoch}.pth")
        torch.save({
            "vae_state_dict": vae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
        }, epoch_path)
        logger.info(f"Saved epoch checkpoint: {epoch_path}")

    # Save final
    final_path = osp.join(args.output_dir, "vae_final.pth")
    torch.save({
        "vae_state_dict": vae.state_dict(),
        "step": global_step,
    }, final_path)
    logger.info(f"Training complete. Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TAESD VAE on Pacman frames")
    parser.add_argument("--vae_pretrained", type=str, default="madebyollin/taesd")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--output_dir", type=str, default="output/vae_pacman")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--buffer_size", type=int, default=200)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
