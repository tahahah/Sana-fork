#!/usr/bin/env python3
"""Does the fine-tuned decoder help GENERATED latents (not just clean ones)?

Generates a frame from the diffusion model (fixed conditioning + noise) and
decodes BOTH the generated latent and the ground-truth latent with the stock
TAESD decoder and the Pacman-fine-tuned decoder. This tells us whether the
decoder upgrade transfers to real gameplay (generated latents) or only flatters
clean encoder latents.
"""
import argparse, os, os.path as osp, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import numpy as np, torch
from PIL import Image, ImageDraw
from diffusion import DPMS
from diffusion.data.datasets.pacman_data import PacmanMapDataset
from diffusers import AutoencoderTiny
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from inference_pacman import load_config, build_inference_model, load_checkpoint_into_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sana_config/512ms/Sana_pacman.yaml")
ap.add_argument("--ckpt", default="output/pacman_latent_v2/checkpoints/latest.pth")
ap.add_argument("--ft_decoder", default="output/vae_decoder_ft/vae_decoder_ft_best.pth")
ap.add_argument("--out", default="output/vae_decoder_ft/generated_decode.png")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--idx", type=int, default=500000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
dev = "cuda"

config = load_config(args.config)
os.makedirs(config.work_dir, exist_ok=True); os.makedirs(config.train.null_embed_root, exist_ok=True)
model, vae = build_inference_model(config, dev)                 # vae = stock TAESD
model = load_checkpoint_into_model(model, args.ckpt, config, dev).eval()

# fine-tuned VAE (same frozen encoder, tuned decoder)
vae_ft = AutoencoderTiny.from_pretrained(config.vae.vae_pretrained).to(dev).eval()
sd = torch.load(args.ft_decoder, map_location="cpu", weights_only=False)["vae_state_dict"]
vae_ft.load_state_dict(sd)

ds = PacmanMapDataset(resolution=config.model.image_size, sequence_length=config.data.sequence_length,
                      load_vae_feat=True, data_dir=list(config.data.data_dir), config=config)
item = ds[args.idx]
obs_lat = item["obs_latent"].unsqueeze(0).to(dev)
y = item["y"].unsqueeze(0).to(dev).float()
tgt_lat = item["img_latent"].unsqueeze(0).to(dev).float()
seq_len = config.data.sequence_length
null_y = torch.zeros(1, 1, seq_len - 1, 5, device=dev)
hw = torch.tensor([[config.model.image_size, config.model.image_size]], dtype=torch.float, device=dev)
ar = torch.tensor([[1.0]], device=dev)
Lc = config.vae.vae_latent_dim; Ls = config.model.image_size // config.vae.vae_downsample_rate
with torch.no_grad():
    obs_latent = model.encode_obs(obs_lat.to(next(model.parameters()).dtype)).float()
mk = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=None, obs_latent=obs_latent)

g = torch.Generator(device=dev).manual_seed(args.seed)
z = torch.randn(1, Lc, Ls, Ls, device=dev, generator=g)
sol = DPMS(model.forward_with_dpmsolver, condition=y, uncondition=null_y, cfg_scale=4.5,
           model_type="flow", model_kwargs=mk, schedule="FLOW")
gen_lat = sol.sample(z, steps=args.steps, order=2, skip_type="time_uniform_flow",
                     method="multistep", flow_shift=config.scheduler.flow_shift).float()

def dec(v, lat):
    with torch.no_grad():
        px = v.decoder(lat).clamp(0, 1)
    return (px[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

def lab(a, t):
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 13], fill=(0, 0, 0)); d.text((3, 2), t, fill=(255, 255, 0)); return np.array(im)

gap = np.full((256, 4, 3), 40, "uint8")
row_gt = np.concatenate([lab(dec(vae, tgt_lat), "GT latent | stock"), gap, lab(dec(vae_ft, tgt_lat), "GT latent | fine-tuned")], 1)
row_gen = np.concatenate([lab(dec(vae, gen_lat), f"GEN {args.steps}st | stock"), gap, lab(dec(vae_ft, gen_lat), f"GEN {args.steps}st | fine-tuned")], 1)
sep = np.full((4, row_gt.shape[1], 3), 40, "uint8")
Image.fromarray(np.concatenate([row_gt, sep, row_gen], 0)).save(args.out)
print("saved", args.out)
