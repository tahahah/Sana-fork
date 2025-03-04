#!/usr/bin/env python
"""
Simple inference script that uses the same checkpoint loading logic as train_pacman.py
"""

import os
import os.path as osp
import sys
import torch
import argparse
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model, get_vae
from diffusion.utils.checkpoint import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference with robust checkpoint loading")
    parser.add_argument("--config", type=str, default="configs/sana_config/512ms/Sana_pacman.yaml", 
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="latest", 
                        help="Path to checkpoint file or 'latest'")
    parser.add_argument("--work_dir", type=str, default="output/debug", 
                        help="Working directory (where checkpoints are stored)")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug output")
    parser.add_argument("--headless", action="store_true", 
                        help="Run without display")
    return parser.parse_args()

def find_checkpoint(checkpoint_path, work_dir):
    """
    Find the checkpoint using the same logic as in train_pacman.py
    """
    print(f"Finding checkpoint: {checkpoint_path}")
    
    # If checkpoint is a full path and exists, use it directly
    if osp.exists(checkpoint_path) and not checkpoint_path == "latest":
        print(f"Using provided checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    # Handle 'latest' or look in work_dir/checkpoints
    ckpt_path = osp.join(work_dir, "checkpoints")
    check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
    
    if checkpoint_path == "latest":
        if check_flag:
            checkpoints = os.listdir(ckpt_path)
            if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                checkpoint_path = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                print(f"Using latest.pth: {checkpoint_path}")
            else:
                checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    checkpoint_path = osp.join(ckpt_path, checkpoints[-1])
                    print(f"Using latest checkpoint by step: {checkpoint_path}")
                else:
                    print("No checkpoints found in the checkpoints directory")
                    return None
        else:
            print("No checkpoints directory found")
            return None
    
    # Final check to ensure checkpoint exists
    if not osp.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    return checkpoint_path

def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = read_config(args.config)
    
    # Set work directory from args
    config.work_dir = args.work_dir
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(args.checkpoint, config.work_dir)
    if checkpoint_path is None:
        print("Failed to find a valid checkpoint")
        return 1
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Build model
    print("Building model...")
    
    # Build model with the same parameters as in train_pacman.py
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "y_norm": True,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.model.in_channels,
        "y_norm_scale_factor": 0.01,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": getattr(config.scheduler, "pred_sigma", True),
        "learn_sigma": getattr(config.scheduler, "learn_sigma", True),
        "caption_channels": config.model.num_classes,
        "model_max_length": config.data.sequence_length-1,
        "seq_length": config.data.sequence_length,
    }
    
    # Load VAE
    print(f"Loading VAE: {config.vae.vae_type} from {config.vae.vae_pretrained}")
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, args.device)
    vae = vae.to(torch.float16)
    
    # Set VAE scaling factor
    if hasattr(vae, 'cfg') and vae.cfg.scaling_factor is None:
        vae.cfg.scaling_factor = config.vae.scale_factor
    
    # Calculate latent size
    latent_size = int(config.model.image_size) // config.vae.vae_downsample_rate
    
    # Build model
    model = build_model(
        config.model.model,
        False,  # grad_checkpointing
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    )
    
    # Prepare null embed path
    null_embed_path = None
    if hasattr(config.train, "null_embed_root"):
        null_embed_path = osp.join(config.train.null_embed_root, f"null_embed_{config.model.num_classes}.pt")
        if not osp.exists(null_embed_path):
            print(f"Warning: Null embed path does not exist: {null_embed_path}")
            null_embed_path = None
    
    # Load checkpoint directly using the same function as in training
    try:
        _, missing, unexpected, _ = load_checkpoint(
            checkpoint_path,
            model,
            load_ema=False,  # We don't need EMA for inference
            null_embed_path=null_embed_path,
        )
        
        print(f"Successfully loaded checkpoint")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        
        # Move model to device and eval mode
        model = model.to(args.device)
        
        # Convert model to fp16 to match input tensors
        if config.model.mixed_precision == 'fp16':
            model = model.to(torch.float16)
        
        model.eval()
        
        # Print model device
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        print("Model loaded successfully!")
        
        # Now you can continue with the rest of your inference code
        # For now, we'll just exit successfully
        return 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
