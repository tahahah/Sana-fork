#!/usr/bin/env python3
"""
Precompute VAE latents for the Pacman dataset.

Loads the local dataset (datasets/pacman_raw), applies the same transforms as the
training dataset (make_square, resize to 256, hflip, rotate 90 CW, ToTensor),
encodes through TAESD VAE, and saves latents [N, 4, 32, 32] + metadata.

Usage:
    python scripts/precompute_vae_latents.py --save-dir ./datasets/pacman_latents
"""

import os
import argparse
import time
import numpy as np
import torch
from PIL import Image
from datasets import load_from_disk
from torchvision import transforms
from diffusers import AutoencoderTiny


def make_square(image):
    width, height = image.size
    max_dim = max(width, height)
    padding = [
        (max_dim - width) // 2,
        (max_dim - height) // 2,
        (max_dim - width + 1) // 2,
        (max_dim - height + 1) // 2,
    ]
    return transforms.functional.pad(image, padding, fill=0, padding_mode="constant")


def convert_to_rgb(img):
    return img.convert("RGB")


def rotate_90_clockwise(img):
    return img.rotate(90, expand=True)


def build_transform(resolution=256):
    return transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Lambda(make_square),
        transforms.Resize(resolution),
        transforms.functional.hflip,
        transforms.Lambda(rotate_90_clockwise),
        transforms.ToTensor(),  # [0,1]
    ])


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for Pacman dataset")
    parser.add_argument("--save-dir", type=str, default="./datasets/pacman_latents")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--vae", type=str, default="madebyollin/taesd")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv("/teamspace/studios/this_studio/personal/Sana-fork/.env")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    vae_dtype = torch.float16

    print(f"Device: {device}, latent dtype: {args.dtype}, VAE dtype: fp16")

    # Load VAE
    print(f"Loading VAE: {args.vae}")
    vae = AutoencoderTiny.from_pretrained(args.vae, torch_dtype=vae_dtype).to(device)
    vae.eval()

    # Build transform
    transform = build_transform(args.resolution)

    # Load local dataset
    local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "pacman_raw")
    print(f"Loading local dataset from {local_path}...")
    ds = load_from_disk(local_path)
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Dataset: {n_total} frames")

    os.makedirs(args.save_dir, exist_ok=True)

    # Process in batches
    all_latents = []
    all_episodes = []
    all_actions = []
    all_dones = []
    total = 0
    start_time = time.time()

    print(f"Processing (batch_size={args.batch_size})...")

    for i in range(0, n_total, args.batch_size):
        end = min(i + args.batch_size, n_total)
        batch = ds[i:end]

        images = torch.stack([transform(img) for img in batch["frame_image"]]).to(device).to(vae_dtype)
        with torch.no_grad():
            latents = vae.encode(images).latents.cpu().to(dtype)

        all_latents.append(latents)
        all_episodes.extend(batch["episode"])
        all_actions.extend(batch["action"])
        all_dones.extend(batch["done"])

        total = end
        elapsed = time.time() - start_time
        fps = total / elapsed
        eta_min = (n_total - total) / fps / 60 if fps > 0 else 0
        print(f"  {total:>8} / {n_total} frames | {fps:.0f} fps | {elapsed:.0f}s | ETA {eta_min:.1f} min")

        # Checkpoint
        if total % args.checkpoint_interval < args.batch_size:
            print(f"  Checkpointing at {total} frames...")
            partial = torch.cat(all_latents, dim=0)
            torch.save({
                "latents": partial,
                "episodes": all_episodes,
                "actions": all_actions,
                "dones": all_dones,
            }, os.path.join(args.save_dir, "pacman_latents.pt"))
            print(f"  Checkpoint saved ({partial.shape[0]} frames)")

    # Final save
    print("Saving final latents...")
    all_latents = torch.cat(all_latents, dim=0)
    save_path = os.path.join(args.save_dir, "pacman_latents.pt")
    torch.save({
        "latents": all_latents,
        "episodes": all_episodes,
        "actions": all_actions,
        "dones": all_dones,
    }, save_path)

    elapsed = time.time() - start_time
    print(f"\nDone! {total} frames in {elapsed:.1f}s ({total/elapsed:.0f} fps)")
    print(f"Latents: {all_latents.shape}, dtype: {all_latents.dtype}")
    print(f"Saved to {save_path}")
    print(f"Storage: {all_latents.element_size() * all_latents.nelement() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
