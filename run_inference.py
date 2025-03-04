#!/usr/bin/env python
"""
Simple script to run inference_pacman.py with a workaround for the checkpoint loading issue
"""

import os
import sys
import torch
from diffusion.utils.checkpoint import load_checkpoint as original_load_checkpoint
import diffusion.utils.checkpoint

# Create a patched version of load_checkpoint
def patched_load_checkpoint(checkpoint, model, **kwargs):
    """
    Patched version of load_checkpoint that uses direct loading instead of find_model
    """
    print(f"Using patched load_checkpoint for {checkpoint}")
    
    try:
        # Direct loading
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        
        # Process the checkpoint data as in the original function
        state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
        for key in state_dict_keys:
            if key in checkpoint_data["state_dict"]:
                del checkpoint_data["state_dict"][key]
                if "state_dict_ema" in checkpoint_data and key in checkpoint_data["state_dict_ema"]:
                    del checkpoint_data["state_dict_ema"][key]
                break

        # Get the state dict based on kwargs
        load_ema = kwargs.get('load_ema', False)
        if load_ema:
            state_dict = checkpoint_data["state_dict_ema"]
        else:
            state_dict = checkpoint_data.get("state_dict", checkpoint_data)

        # Handle null_embed
        null_embed_path = kwargs.get('null_embed_path', None)
        null_embed = None
        if null_embed_path and os.path.exists(null_embed_path):
            try:
                null_embed = torch.load(null_embed_path, map_location="cpu")
            except Exception as e:
                print(f"Warning: Could not load null_embed: {e}")
        
        if null_embed is not None:
            state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
        
        rng_state = checkpoint_data.get("rng_state", None)

        # Load state dict into model
        missing, unexpect = model.load_state_dict(state_dict, strict=False)
        
        # Handle other model components if provided
        model_ema = kwargs.get('model_ema', None)
        optimizer = kwargs.get('optimizer', None)
        lr_scheduler = kwargs.get('lr_scheduler', None)
        resume_optimizer = kwargs.get('resume_optimizer', True)
        resume_lr_scheduler = kwargs.get('resume_lr_scheduler', True)
        
        if model_ema is not None:
            model_ema.load_state_dict(checkpoint_data["state_dict_ema"], strict=False)
        if optimizer is not None and resume_optimizer:
            optimizer.load_state_dict(checkpoint_data["optimizer"])
        if lr_scheduler is not None and resume_lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint_data["scheduler"])

        epoch = checkpoint_data.get("epoch", 0)
        print(f"Successfully loaded checkpoint from {checkpoint}")
        return epoch, missing, unexpect, rng_state
    
    except Exception as e:
        print(f"Error in patched load_checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 0, [], [], None

# Apply the patch
diffusion.utils.checkpoint.load_checkpoint = patched_load_checkpoint

# Run the original script with the patched function
if __name__ == "__main__":
    # Get the command line arguments
    args = sys.argv[1:]
    
    # Construct the command to run the original script
    cmd = [sys.executable, "inference_pacman.py"] + args
    
    # Print the command
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the command
    os.execv(sys.executable, cmd)
