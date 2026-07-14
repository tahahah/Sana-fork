#!/usr/bin/env python3
"""Autoregressive rollout of ONE whole episode — the full game pipeline.

Detects the episode containing --start (bounded by episode id / done flag), seeds
from its first 3 real frame-latents, then generates every subsequent frame,
feeding each generated latent back in as the next observation (true gameplay).
Uses Flow-Euler sampling + the fine-tuned decoder. Saves a GT-vs-generated GIF
and reports per-frame FPS. Error accumulation over a full episode is the point.
"""
import argparse, os, os.path as osp, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import numpy as np, torch
from PIL import Image, ImageDraw
from diffusion import FlowEuler
from diffusion.data.datasets.pacman_data import PacmanMapDataset
from diffusers import AutoencoderTiny
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from inference_pacman import load_config, build_inference_model, load_checkpoint_into_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sana_config/512ms/Sana_pacman.yaml")
ap.add_argument("--ckpt", default="output/pacman_latent_v2/checkpoints/latest.pth")
ap.add_argument("--ft_decoder", default="output/vae_decoder_ft/vae_decoder_ft_best.pth")
ap.add_argument("--out", default="output/rollout")
ap.add_argument("--start", type=int, default=500000)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--cfg", type=float, default=4.5)
ap.add_argument("--max_frames", type=int, default=300, help="cap so a huge episode can't run away")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"

config = load_config(args.config)
os.makedirs(config.work_dir, exist_ok=True); os.makedirs(config.train.null_embed_root, exist_ok=True)
model, vae = build_inference_model(config, dev)
model = load_checkpoint_into_model(model, args.ckpt, config, dev).eval()
vae_ft = AutoencoderTiny.from_pretrained(config.vae.vae_pretrained).to(dev).eval()
vae_ft.load_state_dict(torch.load(args.ft_decoder, map_location="cpu", weights_only=False)["vae_state_dict"])

ds = PacmanMapDataset(resolution=config.model.image_size, sequence_length=config.data.sequence_length,
                      load_vae_feat=True, data_dir=list(config.data.data_dir), config=config)
S = config.data.sequence_length - 1
Lc = config.vae.vae_latent_dim; Ls = config.model.image_size // config.vae.vae_downsample_rate
hw = torch.tensor([[config.model.image_size]*2], dtype=torch.float, device=dev)
ar = torch.tensor([[1.0]], device=dev)
null_y = torch.zeros(1, 1, S, 5, device=dev)
mdtype = next(model.parameters()).dtype

# ---- find the episode span containing `start` ----
eps = ds.latent_episodes; dones = ds.latent_dones; N = len(eps)
ep = eps[args.start]
e_start = args.start
while e_start > 0 and eps[e_start - 1] == ep and not bool(dones[e_start - 1]):
    e_start -= 1
e_end = args.start
while e_end + 1 < N and eps[e_end + 1] == ep and not bool(dones[e_end]):
    e_end += 1
ep_len = e_end - e_start + 1
n_gen = min(ep_len - S, args.max_frames)
print(f"episode {ep}: frames [{e_start}..{e_end}] len={ep_len}; seeding {S}, generating {n_gen}"
      + (" (capped)" if ep_len - S > args.max_frames else ""))

def dec(v, lat):
    with torch.no_grad():
        px = v.decoder(lat).clamp(0, 1)
    return (px[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

buf = [ds.latents[e_start + k].unsqueeze(0).to(dev).float() for k in range(S)]  # oldest->newest
gen_frames, gt_frames, times = [], [], []
for t in range(n_gen):
    idx = e_start + S + t
    y = ds[idx]["y"].unsqueeze(0).to(dev).float()
    obs_cat = torch.cat(buf, dim=1)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        obs_latent = model.encode_obs(obs_cat.to(mdtype)).float()
    mk = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=None, obs_latent=obs_latent)
    g = torch.Generator(device=dev).manual_seed(args.seed + t)
    z = torch.randn(1, Lc, Ls, Ls, device=dev, generator=g)
    gl = FlowEuler(model, condition=y, uncondition=null_y, cfg_scale=args.cfg, model_kwargs=mk).sample(z, steps=args.steps).float()
    px = dec(vae_ft, gl)
    torch.cuda.synchronize(); times.append(time.time() - t0)
    gen_frames.append(px)
    gt_frames.append(dec(vae_ft, ds.latents[idx].unsqueeze(0).to(dev).float()))
    buf = buf[1:] + [gl]
    if t % 20 == 0:
        print(f"  frame {t}/{n_gen} ({times[-1]*1000:.0f} ms)")

avg = sum(times[3:]) / max(1, len(times) - 3)
print(f"\nper-frame (excl. warmup): {avg*1000:.0f} ms -> {1/avg:.1f} FPS  (GPU shared w/ training)")

# side-by-side GIF: GT (top) | GEN (bottom), label header
frames = []
for i, (gt, gn) in enumerate(zip(gt_frames, gen_frames)):
    lab = np.full((14, gt.shape[1], 3), 0, "uint8")
    sep = np.full((4, gt.shape[1], 3), 80, "uint8")
    stack = np.concatenate([lab, gt, sep, gn], 0)
    im = Image.fromarray(stack); d = ImageDraw.Draw(im)
    d.text((3, 1), f"GT (top) vs GEN euler{args.steps} (bottom)  frame {i+1}/{n_gen}", fill=(255, 255, 0))
    frames.append(im)
gif = osp.join(args.out, "episode.gif")
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
# also a static strip of every ~Nth frame for quick glance
stride = max(1, n_gen // 10)
picks = list(range(0, n_gen, stride))[:10]
gap = np.full((256, 2, 3), 40, "uint8")
def rowstrip(frs):
    r = frs[picks[0]]
    for p in picks[1:]:
        r = np.concatenate([r, gap, frs[p]], 1)
    return r
strip = np.concatenate([rowstrip(gt_frames), np.full((6, rowstrip(gt_frames).shape[1], 3), 80, "uint8"), rowstrip(gen_frames)], 0)
Image.fromarray(strip).save(osp.join(args.out, "episode_strip.png"))
print("saved", gif, "and episode_strip.png")
