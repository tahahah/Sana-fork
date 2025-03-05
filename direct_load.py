#!/usr/bin/env python
"""
Minimal script to directly load a checkpoint and apply it to the model
"""

import os
import sys
import torch
import argparse
from diffusion.utils.misc import read_config
from diffusion.model.builder import build_model, get_vae

def parse_args():
    parser = argparse.ArgumentParser(description="Direct checkpoint loading")
    parser.add_argument("--config", type=str, default="configs/sana_config/512ms/Sana_pacman.yaml", 
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = read_config(args.config)
    
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
    
    # Calculate latent size
    latent_size = int(config.model.image_size) // config.vae.vae_downsample_rate
    
    # Build model
    print("Building model...")
    model = build_model(
        config.model.model,
        False,  # grad_checkpointing
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    )
    
    # Load checkpoint directly
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        # Load the checkpoint file directly with torch.load
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        print("Checkpoint loaded successfully")
        
        # Process the checkpoint data
        state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
        for key in state_dict_keys:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]
                if "state_dict_ema" in checkpoint and key in checkpoint["state_dict_ema"]:
                    del checkpoint["state_dict_ema"][key]
                break
        
        # Get the state dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Load state dict into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
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
        return 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
