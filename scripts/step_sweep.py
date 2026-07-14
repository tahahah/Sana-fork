#!/usr/bin/env python3
"""Sampling-step sweep: how few steps can the current model tolerate?

Loads a checkpoint, fixes ONE conditioning batch + ONE initial noise, then
generates the next frame at several step counts / samplers so every panel
differs only by the sampler budget. Saves a labeled grid + a timing table.

Usage:
    PYTHONPATH=. python3 scripts/step_sweep.py \
        --config configs/sana_config/512ms/Sana_pacman.yaml \
        --ckpt output/pacman_latent_v2/checkpoints/latest.pth \
        --out output/step_sweep
"""
import argparse, os, os.path as osp, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import sys
import numpy as np, torch
from PIL import Image, ImageDraw

from diffusion import DPMS, FlowEuler
from diffusion.data.datasets.pacman_data import PacmanMapDataset
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from inference_pacman import load_config, build_inference_model, load_checkpoint_into_model


def label(img_arr, text):
    im = Image.fromarray(img_arr)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 0))
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sana_config/512ms/Sana_pacman.yaml")
    ap.add_argument("--ckpt", default="output/pacman_latent_v2/checkpoints/latest.pth")
    ap.add_argument("--out", default="output/step_sweep")
    ap.add_argument("--idx", type=int, default=500000, help="dataset index for conditioning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    config = load_config(args.config)
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.train.null_embed_root, exist_ok=True)
    model, vae = build_inference_model(config, dev)
    model = load_checkpoint_into_model(model, args.ckpt, config, dev).eval()

    # One fixed conditioning sample (obs_latent + actions) from the map dataset
    ds = PacmanMapDataset(resolution=config.model.image_size,
                          sequence_length=config.data.sequence_length,
                          load_vae_feat=True, data_dir=list(config.data.data_dir),
                          config=config)
    item = ds[args.idx]
    obs_lat = item["obs_latent"].unsqueeze(0).to(dev)          # [1,(seq-1)*4,32,32]
    y = item["y"].unsqueeze(0).to(dev).float()                 # [1,1,seq-1,5]
    tgt_lat = item["img_latent"].unsqueeze(0).to(dev).float()  # ground-truth target latent
    seq_len = config.data.sequence_length
    null_y = torch.zeros(1, 1, seq_len - 1, 5, device=dev)

    hw = torch.tensor([[config.model.image_size, config.model.image_size]], dtype=torch.float, device=dev)
    ar = torch.tensor([[1.0]], device=dev)
    Lc = config.vae.vae_latent_dim
    Ls = config.model.image_size // config.vae.vae_downsample_rate

    with torch.no_grad():
        obs_latent = model.encode_obs(obs_lat.to(next(model.parameters()).dtype)).float()

    model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=None, obs_latent=obs_latent)

    # Fixed initial noise, reused for every run
    g = torch.Generator(device=dev).manual_seed(args.seed)
    z0 = torch.randn(1, Lc, Ls, Ls, device=dev, generator=g)

    def decode(lat):
        with torch.no_grad():
            px = vae.decoder(lat.float()).clamp(0, 1)
        return (px[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    def run_dpm(steps, cfg):
        z = z0.clone()
        sol = DPMS(model.forward_with_dpmsolver, condition=y, uncondition=null_y,
                   cfg_scale=cfg, model_type="flow", model_kwargs=model_kwargs, schedule="FLOW")
        torch.cuda.synchronize(); t = time.time()
        out = sol.sample(z, steps=steps, order=2, skip_type="time_uniform_flow",
                         method="multistep", flow_shift=config.scheduler.flow_shift)
        torch.cuda.synchronize()
        return out, (time.time() - t) * 1000

    def run_euler(steps, cfg):
        z = z0.clone()
        sol = FlowEuler(model, condition=y, uncondition=null_y, cfg_scale=cfg, model_kwargs=model_kwargs)
        torch.cuda.synchronize(); t = time.time()
        out = sol.sample(z, steps=steps)
        torch.cuda.synchronize()
        return out, (time.time() - t) * 1000

    print("NOTE: GPU shared with training jobs — timings are inflated; ratios/quality are the signal.\n")
    panels, timings = [], []
    # reference: VAE recon of the ground-truth target latent (the ceiling)
    panels.append(label(decode(tgt_lat), "GT latent (VAE recon)"))

    def add(fn, steps, cfg, tag):
        try:
            out, ms = fn(steps, cfg)
            panels.append(label(decode(out), f"{tag} {steps}st cfg{cfg} {ms:.0f}ms"))
            print(f"{tag:5s} cfg{cfg} {steps:2d} steps: {ms:7.0f} ms")
        except Exception as e:
            panels.append(label(np.zeros_like(panels[0]), f"{tag} {steps}st cfg{cfg} FAILED"))
            print(f"{tag} cfg{cfg} {steps} steps FAILED: {type(e).__name__}: {e}")

    cfg = 4.5
    for steps in [40, 20, 10, 8, 4, 2]:
        add(run_dpm, steps, cfg, "dpm")
    for steps in [8, 4, 2, 1]:
        add(run_euler, steps, cfg, "euler")
    # no-guidance (cfg=1) via Flow-Euler — half the forwards per step
    for steps in [8, 4, 2]:
        add(run_euler, steps, 1.0, "euler")

    # assemble grid: 4 panels per row
    per_row = 4
    rows = []
    for i in range(0, len(panels), per_row):
        row = panels[i:i + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(panels[0]))
        gap = np.full((panels[0].shape[0], 4, 3), 40, dtype="uint8")
        r = row[0]
        for p in row[1:]:
            r = np.concatenate([r, gap, p], axis=1)
        rows.append(r)
    sep = np.full((4, rows[0].shape[1], 3), 40, dtype="uint8")
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, sep, r], axis=0)
    Image.fromarray(grid).save(osp.join(args.out, "step_sweep.png"))
    print(f"\nsaved {osp.join(args.out, 'step_sweep.png')}  {Image.fromarray(grid).size}")


if __name__ == "__main__":
    main()
