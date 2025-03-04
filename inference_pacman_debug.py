import os
import time
import numpy as np
import torch
import pygame
from PIL import Image
import torchvision.transforms as transforms
from diffusion import DPMS
from diffusion.model.builder import get_vae, vae_encode, vae_decode, build_model
from diffusion.utils.config import SanaConfig
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.data.datasets.pacman_data import convert_to_rgb, make_square, rotate_90_clockwise, to_float16
import pyrallis
from pathlib import Path
from dataclasses import dataclass
from diffusion.utils.misc import read_config

# Constants
FPS = 10  # Target FPS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
ACTION_MAP = {
    pygame.K_LEFT: 0,    # LEFT
    pygame.K_RIGHT: 1,   # RIGHT
    pygame.K_UP: 2,      # UP
    pygame.K_DOWN: 3,    # DOWN
    None: 4             # NO_ACTION
}

@dataclass
class InferenceArgs:
    """
    Arguments for inference
    """
    config: str = "configs/sana_config/512ms/Sana_pacman.yaml"
    checkpoint: str = None
    device: str = "cuda"
    image: str = "scripts/image.jpg"
    debug: bool = False
    x11_display: bool = False  # Enable X11 display for SSH with X forwarding
    dummy_display: bool = False  # Use dummy display driver (for SSH without X11)
    headless: bool = False  # Run without display (for SSH)
    output_dir: str = "output/frames"  # Directory to save frames when in headless mode
    save_frames: bool = False  # Save frames even in non-headless mode

def setup_model(config, checkpoint_path=None, device='cuda', debug=False):
    """
    Set up the model from config and checkpoint
    """
    # Set up model
    print("Building model...")
    
    # ADDED DEBUG: Check checkpoint path
    if checkpoint_path:
        print(f"DEBUG: Checkpoint path: {checkpoint_path}")
        print(f"DEBUG: Checkpoint path exists: {os.path.exists(checkpoint_path)}")
        print(f"DEBUG: Checkpoint path is file: {os.path.isfile(checkpoint_path)}")
    
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
    
    if debug:
        print(f"Model kwargs: {model_kwargs}")
    
    # Load VAE
    print(f"Loading VAE: {config.vae.vae_type} from {config.vae.vae_pretrained}")
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device)
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
    
    if debug:
        print(f"Model structure:")
        print(model)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        
        # ADDED DEBUG: Try direct loading first
        try:
            print("DEBUG: Trying direct loading with torch.load")
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            print(f"DEBUG: Successfully loaded checkpoint with torch.load")
            print(f"DEBUG: Checkpoint keys: {list(checkpoint_data.keys())}")
        except Exception as e:
            print(f"DEBUG: Error with direct loading: {e}")
        
        # Create null_embed_path in the same way as train_pacman.py
        null_embed_path = None
        if os.path.exists("null_embed.pth"):
            null_embed_path = "null_embed.pth"
        
        # Load checkpoint
        try:
            epoch, missing, unexpected, _ = load_checkpoint(
                checkpoint_path, 
                model,
                null_embed_path=null_embed_path
            )
            print(f"Loaded checkpoint from epoch {epoch}")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        except Exception as e:
            print(f"DEBUG: Error in load_checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    # Move model to device and eval mode
    model = model.to(device)
    
    # Convert model to fp16 to match input tensors
    if config.model.mixed_precision == 'fp16':
        model = model.to(torch.float16)
    
    model.eval()
    
    return model, vae, config

def main():
    # Parse arguments
    args = pyrallis.parse(InferenceArgs)
    
    # Print SSH-related settings
    if args.headless:
        print(f"Running in headless mode. Frames will be saved to {args.output_dir}")
    elif args.x11_display:
        print("Using X11 display for SSH forwarding. Make sure you're using 'ssh -X' or 'ssh -Y'")
    elif args.dummy_display:
        print("Using dummy display driver for SSH without X11")
    
    # Load the config file using the utility function
    print(f"Loading config from {args.config}")
    config = read_config(args.config)
    
    # ADDED DEBUG: Check checkpoint path before passing to setup_model
    if args.checkpoint:
        print(f"DEBUG: Checkpoint path from args: {args.checkpoint}")
        print(f"DEBUG: Checkpoint path exists: {os.path.exists(args.checkpoint)}")
        print(f"DEBUG: Checkpoint path is file: {os.path.isfile(args.checkpoint)}")
    
    # Set up model only
    model, vae, config = setup_model(config, args.checkpoint, args.device, args.debug)
    print("Setup model completed")

if __name__ == "__main__":
    main()
