#!/usr/bin/env python3
"""Export PacmanDiffusionModel + VAE decoder to ONNX for TensorRT inference.

Produces:
  output/onnx/diffusion_b2.onnx   — static batch=2 (CFG uncond+cond), fp32
  output/onnx/vae_decoder_b1.onnx — static batch=1, fp32

Usage:
  python scripts/export_trt.py [--config CONFIG] [--ckpt CKPT] [--out OUT_DIR]

Then launch the webui with:
  LD_LIBRARY_PATH=$(python -c "import tensorrt_libs,os;print(os.path.dirname(tensorrt_libs.__file__))"):$LD_LIBRARY_PATH \\
    DPM_TQDM=True python app/webui_pacman.py --use_trt
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch

from inference_pacman import build_inference_model, load_config, load_checkpoint_into_model


def parse_args():
    p = argparse.ArgumentParser(description="Export Pacman model to ONNX for TensorRT")
    p.add_argument("--config", default="configs/sana_config/512ms/Sana_pacman.yaml")
    p.add_argument(
        "--ckpt",
        default="output/pacman_latent_v2/checkpoints/epoch_11_step_282445.pth",
        help="Checkpoint path",
    )
    p.add_argument(
        "--ft_decoder",
        default="output/vae_decoder_ft/vae_decoder_ft_best.pth",
        help="Decoder-only finetuned TAESD",
    )
    p.add_argument("--out", default="output/onnx", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = load_config(args.config)
    if args.ft_decoder and os.path.isfile(args.ft_decoder):
        config.vae.finetuned_decoder = args.ft_decoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Export] Building model on {device}...")
    model, vae = build_inference_model(config, device)
    model = load_checkpoint_into_model(model, args.ckpt, config, device)
    model = model.eval()
    vae = vae.to(device).to(torch.float32).eval()

    S = config.data.sequence_length - 1  # 3 obs frames
    Lc = config.vae.vae_latent_dim       # 4
    Ls = config.model.image_size // config.vae.vae_downsample_rate  # 32
    img_size = config.model.image_size   # 256

    # ── 1. Diffusion model (static batch=2 for CFG) ──
    class DiffusionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, timestep, y, img_hw, aspect_ratio, obs_latent):
            return self.model.forward_with_dpmsolver(
                x, timestep, y,
                data_info={"img_hw": img_hw, "aspect_ratio": aspect_ratio},
                obs_latent=obs_latent,
            )

    B = 2  # CFG: uncond + cond
    wrapper = DiffusionWrapper(model).to(device).eval()
    dummy_x = torch.randn(B, Lc, Ls, Ls, device=device)
    dummy_t = torch.tensor([500.0, 500.0], device=device, dtype=torch.float32)  # float32 — scheduler produces fractional timesteps
    dummy_y = torch.randn(B, 1, S, 5, device=device)
    dummy_hw = torch.tensor([[img_size, img_size]] * B, dtype=torch.float, device=device)
    dummy_ar = torch.tensor([[1.0]] * B, device=device)
    dummy_obs = torch.randn(B, Lc, Ls, Ls, device=device)

    with torch.no_grad():
        out = wrapper(dummy_x, dummy_t, dummy_y, dummy_hw, dummy_ar, dummy_obs)
    assert out.shape == (B, Lc, Ls, Ls), f"Unexpected output shape: {out.shape}"

    diff_path = os.path.join(args.out, "diffusion_b2.onnx")
    print(f"[Export] Exporting diffusion model (batch={B}) to {diff_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t, dummy_y, dummy_hw, dummy_ar, dummy_obs),
        diff_path,
        input_names=["x", "timestep", "y", "img_hw", "aspect_ratio", "obs_latent"],
        output_names=["velocity"],
        opset_version=18,
    )
    print(f"[Export] Diffusion ONNX saved ({os.path.getsize(diff_path) / 1e6:.1f} MB)")

    # ── 2. VAE decoder (static batch=1) ──
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder

        def forward(self, x):
            return self.decoder(x)

    vae_wrapper = VAEDecoderWrapper(vae).to(device).eval()
    dummy_latent = torch.randn(1, Lc, Ls, Ls, device=device)

    with torch.no_grad():
        dec_out = vae_wrapper(dummy_latent)
    assert dec_out.shape == (1, 3, img_size, img_size), f"Unexpected VAE output: {dec_out.shape}"

    vae_path = os.path.join(args.out, "vae_decoder_b1.onnx")
    print(f"[Export] Exporting VAE decoder (batch=1) to {vae_path}...")
    torch.onnx.export(
        vae_wrapper,
        dummy_latent,
        vae_path,
        input_names=["latent"],
        output_names=["image"],
        opset_version=18,
    )
    print(f"[Export] VAE decoder ONNX saved ({os.path.getsize(vae_path) / 1e6:.1f} MB)")

    print("\n[Export] Done. Launch with:")
    trt_libs = os.path.dirname(__import__("tensorrt_libs").__file__)
    print(f"  LD_LIBRARY_PATH={trt_libs}:$LD_LIBRARY_PATH \\")
    print(f"  DPM_TQDM=True python app/webui_pacman.py --use_trt")


if __name__ == "__main__":
    main()
