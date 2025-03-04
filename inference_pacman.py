# Inference script for Pacman model
import os
import time
import numpy as np
import torch
import pygame
from PIL import Image
import torchvision.transforms as transforms
from diffusion import DPMS
from diffusion.model.builder import build_model, get_vae, vae_decode
from diffusion.utils.config import SanaConfig
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.data.datasets.pacman_data import convert_to_rgb, make_square, rotate_90_clockwise, to_float16
import pyrallis
from pathlib import Path
from dataclasses import dataclass

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
class InferenceConfig:
    config_path: str = "configs/sana_config/512ms/Sana_pacman.yaml"
    checkpoint: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image: str = "scripts/image.jpg"

def setup_model(config, checkpoint_path=None, device='cuda'):
    """
    Set up the model from config and checkpoint
    """
    # Set up model
    print("Building model...")
    model = build_model(config.model)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        epoch, missing, unexpected, _ = load_checkpoint(
            checkpoint_path, 
            model,
            null_embed_path="null_embed.pth" if os.path.exists("null_embed.pth") else None
        )
        print(f"Loaded checkpoint from epoch {epoch}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    
    # Load VAE
    print(f"Loading VAE: {config.vae.vae_type} from {config.vae.vae_pretrained}")
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device)
    vae = vae.to(torch.float16)
    
    # Set VAE scaling factor
    if hasattr(vae, 'cfg') and vae.cfg.scaling_factor is None:
        vae.cfg.scaling_factor = config.vae.scale_factor
    
    # Move model to device and eval mode
    model = model.to(device)
    model.eval()
    
    return model, vae, config

def load_initial_frame(image_path, resolution=512):
    """
    Load the initial frame and apply same transformations as in pacman_data.py
    """
    transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Lambda(make_square),
        transforms.Resize(resolution),
        transforms.functional.hflip,
        transforms.Lambda(rotate_90_clockwise),
        transforms.ToTensor(),
        transforms.Lambda(to_float16),
    ])
    
    # Load the image
    image = Image.open(image_path)
    transformed_image = transform(image)
    
    return transformed_image

def process_frame_for_display(frame_tensor):
    """
    Convert a PyTorch tensor to a NumPy array for display
    """
    # Convert to numpy and apply necessary transformations
    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
    frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
    # Rotate back and flip to match original orientation
    frame_pil = Image.fromarray(frame_np)
    frame_pil = frame_pil.rotate(-90, expand=True)
    frame_pil = frame_pil.transpose(Image.FLIP_LEFT_RIGHT)
    frame_np = np.array(frame_pil)
    return frame_np

def one_hot_encode(action, num_classes=5):
    """
    One-hot encode an action
    """
    vector = torch.zeros(num_classes, dtype=torch.float16)
    vector[action] = 1.0
    return vector

def run_pacman_inference(config, checkpoint_path=None, device='cuda', image_path='scripts/image.jpg'):
    """
    Run Pacman model inference loop
    """
    # Set up PyGame for visualization and input handling
    pygame.init()
    resolution = config.model.image_size  # Use resolution from config
    window = pygame.display.set_mode((resolution, resolution))
    pygame.display.set_caption("Pacman Model Inference")
    clock = pygame.time.Clock()
    
    # Set up model and VAE
    model, vae, config = setup_model(config, checkpoint_path, device)
    
    # Initial frame setup
    seq_len = config.data.sequence_length
    print(f"Using sequence length: {seq_len}")
    
    # Create blank (black) frames for initial sequence
    blank_frame = torch.zeros((3, resolution, resolution), dtype=torch.float16, device=device)
    
    # Load the initial frame
    initial_frame = load_initial_frame(image_path, resolution=resolution)
    initial_frame = initial_frame.to(device)
    
    # Set up initial sequence with black frames + initial frame
    frames = [blank_frame] * (seq_len - 1) + [initial_frame]
    
    # Set up actions (initially all NO_ACTION)
    actions = [one_hot_encode(4).to(device)] * (seq_len - 1)  # No action for all initial frames
    
    # Set up initial input tensors for model
    frames_tensor = torch.stack(frames)  # [seq_len, C, H, W]
    
    # Main loop
    running = True
    current_action = 4  # Start with NO_ACTION
    
    print("Starting inference loop...")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in ACTION_MAP:
                    current_action = ACTION_MAP[event.key]
            elif event.type == pygame.KEYUP:
                if event.key in ACTION_MAP:
                    current_action = 4  # NO_ACTION when key is released
        
        # Add current action to actions list and remove oldest
        actions.append(one_hot_encode(current_action).to(device))
        actions = actions[1:]
        
        # Prepare inputs for model
        # 1. Flatten frames for observation input
        obs_frames = frames_tensor[:-1].view(-1, resolution, resolution)  # [(seq_len-1)*C, H, W]
        
        # 2. Add noise to observation frames
        noise = torch.randn_like(obs_frames)
        noise_scale = 0.3 * torch.rand(1).item()
        noisy_obs = obs_frames + noise_scale * noise
        
        # 3. Get the last frame as target image
        img = frames_tensor[-1]  # [C, H, W]
        
        # 4. Set up actions tensor for model
        actions_tensor = torch.stack(actions).unsqueeze(0)  # [1, seq_len-1, 5]
        action_masks = torch.ones(1, seq_len-1, device=device)  # [1, seq_len-1]
        
        # 5. Generate noise for denoising
        z = torch.randn_like(img.unsqueeze(0))  # [1, C, H, W]
        
        # 6. Set up model kwargs
        hw = torch.tensor([[resolution, resolution]], dtype=torch.float, device=device)
        ar = torch.tensor([[1.0]], device=device)
        
        # Null action for classifier-free guidance
        null_action = torch.zeros(1, 1, seq_len-1, 5, device=device)
        null_action_mask = torch.ones(1, seq_len-1, device=device)
        
        model_kwargs = {
            'data_info': {'img_hw': hw, 'aspect_ratio': ar},
            'mask': None,
            'obs': noisy_obs.unsqueeze(0),  # Add batch dimension
        }
        
        # Run sampling
        with torch.no_grad():
            # Use DPM-Solver for sampling
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=actions_tensor.unsqueeze(1),  # [1, 1, seq_len-1, 5]
                uncondition=null_action,
                cfg_scale=4.5,
                model_kwargs=model_kwargs,
                model_type="flow",
                schedule="FLOW",
            )
            
            # Sample with reduced steps for real-time performance
            denoised = dpm_solver.sample(
                z,
                steps=20,  # Reduced from 40 for real-time
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=config.scheduler.flow_shift,
            )
            
            # Decode the latent using VAE
            denoised = denoised.to(torch.float16)
            samples = vae_decode(config.vae.vae_type, vae, denoised)
            
            # Process for display
            output_frame = samples[0]  # Remove batch dimension
            
        # Convert tensor to numpy for display
        display_frame = process_frame_for_display(output_frame)
        
        # Update display
        pygame_surface = pygame.surfarray.make_surface(display_frame)
        window.blit(pygame_surface, (0, 0))
        pygame.display.flip()
        
        # Update frames for next iteration
        frames.append(output_frame.cpu())
        frames = frames[1:]
        frames_tensor = torch.stack(frames).to(device)
        
        # Cap the framerate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()

if __name__ == "__main__":
    # Parse arguments using pyrallis
    inference_cfg = pyrallis.parse(InferenceConfig)
    
    # Load the model config
    print(f"Loading config from {inference_cfg.config_path}")
    config = pyrallis.parse(SanaConfig, Path(inference_cfg.config_path))
    
    run_pacman_inference(
        config=config,
        checkpoint_path=inference_cfg.checkpoint,
        device=inference_cfg.device,
        image_path=inference_cfg.image
    )
