#!/usr/bin/env python3
"""
Standalone inference script for the Pacman diffusion model.

Loads a checkpoint from HF (or local), builds the model + VAE, samples
a batch from the Pacman dataset, runs DPM-Solver / Flow-Euler sampling,
and saves generated frames as images.

Usage:
    python scripts/inference_pacman.py \
        --config configs/sana_config/512ms/Sana_pacman.yaml \
        --model_path hf://Tahahah/pacman-sana-3.2m-taesd/checkpoints/epoch_33_step_440001.pth

    # Or with a local checkpoint:
    python scripts/inference_pacman.py \
        --config configs/sana_config/512ms/Sana_pacman.yaml \
        --model_path /path/to/checkpoint.pth

    # Custom sampler / steps / output dir:
    python scripts/inference_pacman.py \
        --config configs/sana_config/512ms/Sana_pacman.yaml \
        --model_path hf://Tahahah/pacman-sana-3.2m-taesd/checkpoints/epoch_33_step_440001.pth \
        --sampler flow_dpm-solver --steps 40 --output_dir output/inference
"""

import argparse
import os
import os.path as osp
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from termcolor import colored

warnings.filterwarnings("ignore")

# Load .env for HF_TOKEN
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from diffusion import DPMS, FlowEuler
from diffusion.data.builder import build_dataset
from diffusion.model.builder import build_model, get_vae
from diffusion.utils.config import SanaConfig

import pyrallis


def parse_args():
    parser = argparse.ArgumentParser(description="Pacman diffusion inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path (local or hf://repo_id/path)")
    parser.add_argument("--output_dir", type=str, default="output/inference", help="Directory to save images")
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["flow_dpm-solver", "dpm-solver", "flow_euler"],
                        help="Sampling algorithm (default: from config)")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps (default: 40 for DPM, 28 for Euler)")
    parser.add_argument("--cfg_scale", type=float, default=4.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    return parser.parse_args()


def load_config(config_path):
    """Load SanaConfig from YAML."""
    config = pyrallis.load(SanaConfig, open(config_path))
    return config


def download_checkpoint(model_path):
    """Download checkpoint from HF if needed, return local path."""
    if osp.exists(model_path):
        return model_path

    if model_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        segs = model_path.replace("hf://", "").split("/")
        repo_id = "/".join(segs[:2])
        filename = "/".join(segs[2:])

        token = os.environ.get("HF_TOKEN")
        print(colored(f"[HF] Downloading {repo_id}/{filename}", "cyan"))
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
        )
        return local_path

    raise FileNotFoundError(f"Could not find checkpoint at {model_path}")


def build_inference_model(config, device):
    """Build the PacmanDiffusionModel and VAE."""
    image_size = config.model.image_size
    latent_size = image_size // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma

    # Build VAE (with fine-tuned decoder if configured)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device,
                  finetuned_decoder=getattr(config.vae, "finetuned_decoder", None)).to(torch.float32)
    print(colored(f"[VAE] {config.vae.vae_type} loaded", "green"))

    # Build model
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "y_norm": True,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.model.in_channels,
        "y_norm_scale_factor": 1.0,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "caption_channels": config.model.num_classes,
        "model_max_length": config.data.sequence_length - 1,
        "seq_length": config.data.sequence_length,
        "vae": vae,
        "accelerator": None,
    }

    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).eval().to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(colored(f"[Model] {config.model.model}: {total_params:.2f}M params", "green"))

    return model, vae


def load_checkpoint_into_model(model, checkpoint_path, config, device):
    """Load checkpoint weights into model."""
    print(colored(f"[Checkpoint] Loading from {checkpoint_path}", "cyan"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove pos_embed keys that don't match (checkpoint may have different spatial size)
    for key in ["pos_embed", "base_model.pos_embed", "model.pos_embed", "sana.pos_embed"]:
        if key in state_dict:
            del state_dict[key]

    # Drop VAE weights baked into the checkpoint — the VAE (incl. the fine-tuned
    # decoder) is loaded separately via get_vae; otherwise these stock weights
    # overwrite the fine-tuned decoder on load (blobby output).
    for k in [k for k in state_dict if k.startswith("vae.")]:
        del state_dict[k]

    # Load null_embed if available
    null_embed_root = config.train.null_embed_root
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    null_embed_path = osp.join(
        null_embed_root,
        f"null_embed_diffusers_{config.vae.vae_type}_{latent_size}.pth",
    )

    if osp.exists(null_embed_path):
        null_embed = torch.load(null_embed_path, map_location="cpu")
        if null_embed is not None and "y_embedder.y_embedding" not in state_dict:
            state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
            print(colored("[Checkpoint] Loaded null_embed", "green"))
    else:
        # Try downloading from HF
        try:
            import huggingface_hub
            token = os.environ.get("HF_TOKEN")
            if token:
                huggingface_hub.login(token=token)
                null_embed_filename = osp.basename(null_embed_path)
                null_embed_file = huggingface_hub.hf_hub_download(
                    repo_id="Tahahah/pacman-sana-3.2m-taesd-v2",
                    filename=f"pretrained_models/{null_embed_filename}",
                    repo_type="model",
                    token=token,
                )
                null_embed = torch.load(null_embed_file, map_location="cpu")
                state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
                print(colored("[Checkpoint] Loaded null_embed from HF", "green"))
        except Exception as e:
            print(colored(f"[Checkpoint] Could not load null_embed: {e}", "yellow"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(colored(f"[Checkpoint] Missing keys: {missing}", "yellow"))
    if unexpected:
        print(colored(f"[Checkpoint] Unexpected keys: {unexpected}", "yellow"))

    return model


def get_sample_batch(config, device, num_samples):
    """Get a batch of samples from the Pacman dataset for conditioning."""
    image_size = config.model.image_size
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        vae_downsample_rate=config.vae.vae_downsample_rate,
        vae=None,
    )

    # The dataset is an IterableDataset — collect samples
    samples = []
    iterator = iter(dataset)
    for _ in range(num_samples):
        try:
            samples.append(next(iterator))
        except StopIteration:
            break

    if not samples:
        raise RuntimeError("Could not get any samples from the dataset")

    # Stack into batch — obs/target are precomputed latents (new pipeline)
    obs_latent = torch.stack([s["obs_latent"] for s in samples]).to(device)
    img_latent = torch.stack([s["img_latent"] for s in samples]).to(device)
    actions = torch.stack([s["y"] for s in samples]).to(device)
    action_masks = torch.stack([s["y_mask"] for s in samples]).to(device)

    return {
        "obs_latent": obs_latent,
        "img_latent": img_latent,
        "actions": actions,
        "action_masks": action_masks,
    }


def run_sampling(model, vae, batch, config, args, device):
    """Run latent diffusion sampling and return generated images."""
    obs_latents = batch["obs_latent"]
    actions = batch["actions"]
    batch_size = obs_latents.shape[0]
    seq_len = config.data.sequence_length

    # Shape info
    hw = torch.tensor(
        [[config.model.image_size, config.model.image_size]],
        dtype=torch.float, device=device,
    ).repeat(batch_size, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(batch_size, 1)

    # Null action for classifier-free guidance
    # Null action for classifier-free guidance [B, 1, seq_len-1, 5]
    null_action = torch.zeros(batch_size, 1, seq_len - 1, 5, device=device)
    null_action_mask = torch.ones(batch_size, 1, seq_len - 1, device=device)

    # Actions from dataset are [B, 1, seq_len-1, 5] — no unsqueeze needed

    # Merge precomputed obs latents through the history encoder (no runtime VAE)
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
        obs_latent = model.encode_obs(obs_latents.to(device).to(model_dtype)).float()

    # Latent dimensions
    latent_channels = config.vae.vae_latent_dim  # 4
    latent_size = config.model.image_size // config.vae.vae_downsample_rate  # 8

    # Initial latent noise
    z = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)
    print(colored(f"[Sampler] Initial latent shape: {z.shape}", "cyan"))

    model_kwargs = dict(
        data_info={"img_hw": hw, "aspect_ratio": ar},
        mask=None,
        obs_latent=obs_latent,
    )

    sampler = args.sampler or config.scheduler.vis_sampler

    if sampler == "flow_dpm-solver":
        steps = args.steps or config.scheduler.vis_sampler_steps
        print(colored(f"[Sampler] flow_dpm-solver, {steps} steps", "cyan"))
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=actions,
            uncondition=null_action,
            cfg_scale=args.cfg_scale,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        denoised = dpm_solver.sample(
            z,
            steps=steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=config.scheduler.flow_shift,
        )
    elif sampler == "dpm-solver":
        steps = args.steps or config.scheduler.vis_sampler_steps
        print(colored(f"[Sampler] dpm-solver, {steps} steps", "cyan"))
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=actions,
            uncondition=null_action,
            cfg_scale=args.cfg_scale,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        denoised = dpm_solver.sample(
            z,
            steps=steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=config.scheduler.flow_shift,
        )
    elif sampler == "flow_euler":
        steps = args.steps or config.scheduler.vis_sampler_steps
        print(colored(f"[Sampler] flow_euler, {steps} steps", "cyan"))
        flow_solver = FlowEuler(
            model,
            condition=actions,
            uncondition=null_action,
            cfg_scale=args.cfg_scale,
            model_kwargs=model_kwargs,
        )
        denoised = flow_solver.sample(z, steps=steps)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    print(colored(f"[Sampler] Output latent shape: {denoised.shape}", "green"))

    # VAE decode latent to pixel image (once, after sampling)
    with torch.no_grad():
        pixel_images = vae.decoder(denoised.float())  # [B, 3, 256, 256] in [0,1]

    action_names = ["LEFT", "RIGHT", "UP", "DOWN", "NO_ACTION"]

    images = []
    for idx in range(pixel_images.shape[0]):
        sample = (
            torch.clamp(255 * pixel_images[idx], 0, 255)
            .permute(1, 2, 0)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        image = Image.fromarray(sample)

        # Decode action sequence for this sample
        action_seq = []
        for i in range(min(seq_len - 1, actions.shape[3])):
            action_idx = actions[idx, 0, 0, i].argmax().item()
            action_seq.append(action_names[action_idx])
        action_str = " -> ".join(action_seq)

        images.append({
            "image": image,
            "actions": action_str,
        })

    return images


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    print(colored(f"[Config] Loading from {args.config}", "cyan"))
    config = load_config(args.config)

    # Create work_dir so the model's logger can write train_log.log
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(config.train.null_embed_root, exist_ok=True)
    # Download checkpoint if needed
    checkpoint_path = download_checkpoint(args.model_path)

    # Build model + VAE
    model, vae = build_inference_model(config, device)

    # Load checkpoint weights
    model = load_checkpoint_into_model(model, checkpoint_path, config, device)
    model = model.to(torch.float16)

    # Get conditioning batch from dataset
    print(colored(f"[Data] Fetching {args.num_samples} samples from dataset", "cyan"))
    batch = get_sample_batch(config, device, args.num_samples)
    print(colored(f"[Data] obs_latent: {batch['obs_latent'].shape}, img_latent: {batch['img_latent'].shape}", "green"))

    # Run sampling
    print(colored("[Inference] Starting sampling...", "cyan"))
    with torch.inference_mode():
        results = run_sampling(model, vae, batch, config, args, device)

    # Save images
    os.makedirs(args.output_dir, exist_ok=True)
    for i, result in enumerate(results):
        filename = osp.join(args.output_dir, f"sample_{i:03d}.png")
        result["image"].save(filename)
        print(colored(f"[Saved] {filename} (actions: {result['actions']})", "green"))

    # Also save a grid
    if len(results) > 1:
        from torchvision.utils import make_grid
        grid_tensors = []
        for r in results:
            t = torch.from_numpy(np.array(r["image"])).permute(2, 0, 1) / 255.0
            grid_tensors.append(t)
        grid = make_grid(grid_tensors, nrow=min(4, len(grid_tensors)))
        grid_path = osp.join(args.output_dir, "grid.png")
        from torchvision.utils import save_image
        save_image(grid, grid_path)
        print(colored(f"[Saved] {grid_path} (grid of {len(results)} images)", "green"))

    print(colored(f"\nDone! {len(results)} images saved to {args.output_dir}", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
