"""Decoder-only fine-tuning of TAESD on Pacman frames.

WHY decoder-only: the encoder is FROZEN, so the latents it produces are unchanged.
That keeps every precomputed latent valid and keeps the diffusion model's target
distribution intact — we only swap in a better *renderer* for the same latents.
Fine-tuning the encoder would invalidate the latents and break the diffusion run.

Loss is crispness-oriented, not plain MSE: MSE is dominated by the ~86% black
background, so it barely penalizes fuzzy walls/dots. We use a bright-pixel-weighted
L1 plus an edge (gradient) term to push sharp, saturated game elements.

NOTE: the encoder's 8x downsampling has already discarded most dots at encode time;
decoder fine-tuning recovers wall/ghost/Pac-Man sharpness and color, but cannot
resurrect information the encoder threw away. (That needs option 1 — less downsampling.)

Usage:
    PYTHONPATH=. python3 train_scripts/finetune_taesd_decoder.py \
        --steps 8000 --batch_size 32 --lr 1e-4 \
        --output_dir output/vae_decoder_ft
"""

import argparse, os, os.path as osp, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_from_disk
from diffusers import AutoencoderTiny
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - [DEC-FT] - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def make_square(image):
    w, h = image.size; m = max(w, h)
    return transforms.functional.pad(
        image, [(m - w) // 2, (m - h) // 2, (m - w + 1) // 2, (m - h + 1) // 2], fill=0)

def build_transform(res=256):
    return transforms.Compose([
        transforms.Lambda(lambda i: i.convert("RGB")),
        transforms.Lambda(make_square),
        transforms.Resize(res),
        transforms.functional.hflip,
        transforms.Lambda(lambda i: i.rotate(90, expand=True)),
        transforms.ToTensor(),  # [0,1]
    ])


class PacmanFrames(Dataset):
    """Map-style random access over the local raw frames (same transform as training)."""
    def __init__(self, root, res=256):
        self.ds = load_from_disk(root)
        self.tf = build_transform(res)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        return self.tf(self.ds[i]["frame_image"])


def crispness_loss(recon, target, bright_thr=0.15, bright_w=9.0, edge_w=0.5):
    """Bright-pixel-weighted L1 + edge (gradient) matching."""
    bright = target.amax(dim=1, keepdim=True)                 # [B,1,H,W]
    w = 1.0 + bright_w * (bright > bright_thr).float()        # lit pixels weighted (1+9)=10x
    l1 = (w * (recon - target).abs()).mean()
    def grad(x):
        return (x[..., :, 1:] - x[..., :, :-1]).abs(), (x[..., 1:, :] - x[..., :-1, :]).abs()
    rgx, rgy = grad(recon); tgx, tgy = grad(target)
    edge = (rgx - tgx).abs().mean() + (rgy - tgy).abs().mean()
    return l1 + edge_w * edge, l1.item(), edge.item()


@torch.no_grad()
def save_recon_grid(vae, eval_imgs, path):
    vae.eval()
    rows = []
    for img in eval_imgs:
        x = img.unsqueeze(0).cuda()
        rec = vae.decode(vae.encode(x).latents).sample.clamp(0, 1)
        o = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        r = (rec[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        gap = np.zeros((o.shape[0], 4, 3), dtype="uint8")
        rows.append(np.concatenate([o, gap, r], axis=1))
    sep = np.full((4, rows[0].shape[1], 3), 40, dtype="uint8")
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, sep, r], axis=0)
    Image.fromarray(grid).save(path)
    vae.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets/pacman_raw")
    ap.add_argument("--vae_pretrained", default="madebyollin/taesd")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--output_dir", default="output/vae_decoder_ft")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--init_from", default=None, help="resume: load a prior fine-tuned vae_state_dict")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    vae = AutoencoderTiny.from_pretrained(args.vae_pretrained).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)["vae_state_dict"]
        vae.load_state_dict(sd)
        logger.info(f"Initialized decoder from {args.init_from}")
    # FREEZE ENCODER — latents must stay identical to the precomputed / diffusion targets.
    for p in vae.encoder.parameters():
        p.requires_grad = False
    for p in vae.decoder.parameters():
        p.requires_grad = True
    vae.encoder.eval()  # freeze norm stats too
    vae.decoder.train()
    n_train = sum(p.numel() for p in vae.decoder.parameters() if p.requires_grad)
    logger.info(f"Decoder trainable params: {n_train/1e6:.2f}M (encoder frozen)")

    ds = PacmanFrames(args.data_root, args.resolution)
    logger.info(f"Dataset frames: {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.num_workers > 0)

    # Fixed eval frames for before/after comparison
    eval_imgs = [ds[i] for i in [2000, 300000, 700000, 1100000]]
    save_recon_grid(vae, eval_imgs, osp.join(args.output_dir, "recon_step0.png"))
    logger.info("Saved baseline recon (step 0)")

    opt = torch.optim.AdamW(vae.decoder.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")

    step = 0; t0 = time.time(); best = float("inf")
    while step < args.steps:
        for x in dl:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z = vae.encoder(x)                       # frozen encoder
            with torch.amp.autocast("cuda"):
                recon = vae.decoder(z).clamp(0, 1)
                loss, l1, edge = crispness_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1

            if step % args.log_interval == 0:
                sps = step / (time.time() - t0)
                logger.info(f"step {step}/{args.steps} | loss {loss.item():.4f} "
                            f"(l1 {l1:.4f}, edge {edge:.4f}) | lr {sched.get_last_lr()[0]:.2e} | {sps:.1f} it/s")
            if step % args.save_every == 0 or step == args.steps:
                save_recon_grid(vae, eval_imgs, osp.join(args.output_dir, f"recon_step{step}.png"))
                torch.save({"vae_state_dict": vae.state_dict(), "step": step},
                           osp.join(args.output_dir, "vae_decoder_ft_latest.pth"))
                if loss.item() < best:
                    best = loss.item()
                    torch.save({"vae_state_dict": vae.state_dict(), "step": step},
                               osp.join(args.output_dir, "vae_decoder_ft_best.pth"))
                logger.info(f"  saved recon + checkpoint @ step {step}")
            if step >= args.steps:
                break
    logger.info("Done.")


if __name__ == "__main__":
    main()
