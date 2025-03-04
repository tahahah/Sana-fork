# Inference script for Pacman model
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
    config: str = "configs/sana_config/512ms/Sana_pacman.yaml"
    checkpoint: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image: str = "scripts/image.jpg"
    debug: bool = False

def setup_model(config, checkpoint_path=None, device='cuda', debug=False):
    """
    Set up the model from config and checkpoint
    """
    # Set up model
    print("Building model...")
    # The model config needs a 'type' key for the registry
    model_config = dict(type=config.model.model)
    # Add all other config parameters
    for key, value in vars(config.model).items():
        if key != 'model':  # Skip the model name as we've already used it as 'type'
            model_config[key] = value
    
    # Explicitly set the sequence length to match what's in the config
    model_config['seq_length'] = config.data.sequence_length
    
    if debug:
        print(f"Model config: {model_config}")
    
    model = build_model(model_config)
    
    if debug:
        print(f"Model structure:")
        print(model)
    
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
    
    # Convert model to fp16 to match input tensors
    if config.model.mixed_precision == 'fp16':
        model = model.to(torch.float16)
    
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
        # Don't convert to float16 here, we'll handle precision in the main function
    ])
    
    # Load the image
    try:
        image = Image.open(image_path)
        transformed_image = transform(image)
        return transformed_image  # Return as float32 tensor
    except FileNotFoundError:
        print(f"Warning: Image file {image_path} not found. Using a blank image instead.")
        # Return a blank (black) image if file not found
        return torch.zeros((3, resolution, resolution))

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

def one_hot_encode(action, num_classes=5, dtype=torch.float16):
    """
    One-hot encode an action
    """
    vector = torch.zeros(num_classes, dtype=dtype)
    vector[action] = 1.0
    return vector

def run_pacman_inference(config, args):
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
    model, vae, config = setup_model(config, args.checkpoint, args.device, args.debug)
    
    # Initial frame setup
    seq_len = config.data.sequence_length
    print(f"Using sequence length: {seq_len}")
    
    # Determine precision based on config
    dtype = torch.float16 if config.model.mixed_precision == 'fp16' else torch.float32
    print(f"Using precision: {dtype}")
    
    # Create blank (black) frames for initial sequence
    blank_frame = torch.zeros((3, resolution, resolution), dtype=dtype, device=args.device)
    
    # Load the initial frame
    initial_frame = load_initial_frame(args.image, resolution=resolution)
    initial_frame = initial_frame.to(dtype).to(args.device)
    
    # Set up initial sequence with black frames + initial frame
    frames = [blank_frame] * (seq_len - 1) + [initial_frame]
    
    # Set up actions (initially all NO_ACTION)
    actions = [one_hot_encode(4, dtype=dtype).to(args.device)] * (seq_len - 1)  # No action for all initial frames
    
    # Set up initial input tensors for model
    frames_tensor = torch.stack(frames)  # [seq_len, C, H, W]
    
    # Main loop
    running = True
    current_action = 4  # Start with NO_ACTION
    
    print("Starting inference loop...")
    
    try:
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
            actions.append(one_hot_encode(current_action, dtype=dtype).to(args.device))
            actions = actions[1:]
            
            # Prepare inputs for model
            # 1. Get observation frames (all frames except the last one)
            obs_frames = frames_tensor[:-1]  # [seq_len-1, C, H, W]
            
            # Debug prints
            if args.debug:
                print(f"Frames tensor shape: {frames_tensor.shape}")
                print(f"Observation frames shape before encoding: {obs_frames.shape}")
            
            # 2. Encode all frames with VAE to get latents
            with torch.no_grad():
                # Process each frame individually through the VAE
                encoded_obs_frames = []
                
                if args.debug:
                    print(f"VAE type: {config.vae.vae_type}")
                    print(f"VAE device: {next(vae.parameters()).device}")
                    print(f"VAE dtype: {next(vae.parameters()).dtype}")
                
                for i in range(obs_frames.shape[0]):
                    # Add batch dimension for VAE
                    frame = obs_frames[i].unsqueeze(0)  # [1, C, H, W]
                    # Encode with VAE
                    encoded = vae_encode(config.vae.vae_type, vae, frame, args.device, sample_posterior=False)
                    if args.debug and i == 0:  # Only print for the first frame
                        print(f"Encoded latent shape: {encoded.shape}")
                        print(f"Encoded latent dtype: {encoded.dtype}")
                    encoded_obs_frames.append(encoded.squeeze(0))  # Remove batch dim
                
                # Stack encoded frames
                encoded_obs = torch.stack(encoded_obs_frames)  # [seq_len-1, 4, h, w]
                
                # Reshape to match expected input format for history encoder
                # The history encoder expects [B, C*seq_len, h, w]
                encoded_obs = encoded_obs.permute(1, 0, 2, 3)  # [4, seq_len-1, h, w]
                latent_h, latent_w = encoded_obs.shape[2], encoded_obs.shape[3]
                encoded_obs = encoded_obs.reshape(4 * (seq_len - 1), latent_h, latent_w)  # [4*(seq_len-1), h, w]
                encoded_obs = encoded_obs.unsqueeze(0)  # Add batch dimension [1, 4*(seq_len-1), h, w]
            
            # Debug prints
            if args.debug:
                print(f"Encoded observation shape: {encoded_obs.shape}")
                print(f"Expected shape: [1, {4 * (seq_len - 1)}, {latent_h}, {latent_w}]")
                print(f"Encoded observation dtype: {encoded_obs.dtype}")
            
            # 3. Add noise to encoded observation
            # noise = torch.randn_like(encoded_obs)
            # noise_scale = 0.3 * torch.rand(1).item()
            # noisy_obs = encoded_obs + noise_scale * noise
            
            # 4. Get the last frame and encode it
            img = frames_tensor[-1].unsqueeze(0)  # Add batch dim [1, C, H, W]
            with torch.no_grad():
                encoded_img = vae_encode(config.vae.vae_type, vae, img, args.device, sample_posterior=False)
                encoded_img = encoded_img.squeeze(0)  # [4, h, w]
            
            # 5. Set up actions tensor for model
            actions_tensor = torch.stack(actions).unsqueeze(0)  # [1, seq_len-1, 5]
            action_masks = torch.ones(1, seq_len-1, device=args.device, dtype=dtype)  # [1, seq_len-1]
            
            # 6. Generate noise for denoising
            z = torch.randn_like(encoded_img.unsqueeze(0))  # [1, 4, h, w]
            
            # 7. Set up model kwargs
            hw = torch.tensor([[resolution, resolution]], dtype=dtype, device=args.device)
            ar = torch.tensor([[1.0]], dtype=dtype, device=args.device)
            
            # Null action for classifier-free guidance
            null_action = torch.zeros(1, 1, seq_len-1, 5, device=args.device, dtype=dtype)
            null_action_mask = torch.ones(1, seq_len-1, device=args.device, dtype=dtype)
            
            model_kwargs = {
                'data_info': {'img_hw': hw, 'aspect_ratio': ar},
                'mask': None,
                'obs': encoded_obs,  # [1, 4*(seq_len-1), h, w]
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
                denoised = denoised.to(dtype)
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
            frames_tensor = torch.stack(frames).to(args.device)
            
            # Cap the framerate
            clock.tick(FPS)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        pygame.quit()

def main():
    # Parse arguments
    args = pyrallis.parse(InferenceArgs)
    
    # Load the config file using the utility function
    print(f"Loading config from {args.config}")
    config = read_config(args.config)
    
    # Run inference
    run_pacman_inference(config, args)

if __name__ == "__main__":
    main()
