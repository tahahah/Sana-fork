# Inference script for Pacman model
import os
import os.path as osp
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
    headless: bool = False  # Run without display (for SSH)
    output_dir: str = "output/frames"  # Directory to save frames when in headless mode
    save_frames: bool = False  # Save frames even in non-headless mode

def setup_model(config, checkpoint_path=None, device='cuda', debug=False):
    """
    Set up the model from config and checkpoint
    """
    # Set up model
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
    elif config.model.resume_from is not None and config.model.resume_from.get("checkpoint", "") == "latest":
        ckpt_path = osp.join(config.train.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
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

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Load the checkpoint file directly with torch.load
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
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
    # Make sure the tensor is on CPU before converting to numpy
    if frame_tensor.device.type != 'cpu':
        frame_tensor = frame_tensor.detach().cpu()
    
    # Convert to numpy and apply necessary transformations
    frame_np = frame_tensor.permute(1, 2, 0).numpy()
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
    if not args.headless:
        # Configure for X11 forwarding if needed
        if args.x11_display:
            # Set SDL to use X11
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            print("Using X11 display for SSH forwarding")
        
        pygame.init()
        resolution = config.model.image_size  # Use resolution from config
        window = pygame.display.set_mode((resolution, resolution))
        pygame.display.set_caption("Pacman Model Inference")
        clock = pygame.time.Clock()
    
    # Set up model and VAE
    model, vae, config = setup_model(config, args.checkpoint, args.device, args.debug)
    
    # Get VAE device for later use
    vae_device = next(vae.parameters()).device
    model_device = next(model.parameters()).device
    if args.debug:
        print(f"Model device: {model_device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"VAE device: {vae_device}")
        print(f"VAE dtype: {next(vae.parameters()).dtype}")
    
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
    
    if args.debug:
        print(f"Blank frame device: {blank_frame.device}, dtype: {blank_frame.dtype}")
        print(f"Initial frame device: {initial_frame.device}, dtype: {initial_frame.dtype}")
    
    # Set up initial sequence with black frames + initial frame
    frames = [blank_frame] * (seq_len - 1) + [initial_frame]
    
    # Set up actions (initially all NO_ACTION)
    actions = [one_hot_encode(4, dtype=dtype).to(args.device)] * (seq_len - 1)  # No action for all initial frames
    
    # Set up initial input tensors for model
    # Make sure all frames are on the same device before stacking
    frames_tensor = torch.stack([frame.to(args.device) for frame in frames])
    
    if args.debug:
        print(f"Frames tensor shape: {frames_tensor.shape}")
        print(f"Frames tensor device: {frames_tensor.device}, dtype: {frames_tensor.dtype}")
    
    # Main loop
    running = True
    current_action = 4  # Start with NO_ACTION
    frame_count = 0
    
    print("Starting inference loop...")
    
    try:
        while running:
            # Handle events
            if not args.headless:
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
                
                for i in range(obs_frames.shape[0]):
                    # Add batch dimension for VAE
                    frame = obs_frames[i].unsqueeze(0)  # [1, C, H, W]
                    # Ensure frame is on the same device as VAE
                    frame = frame.to(vae_device)
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
            # Ensure image is on the same device as VAE
            img = img.to(vae_device)
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
                
                if args.debug:
                    print(f"Output frame shape: {output_frame.shape}")
                    print(f"Output frame device: {output_frame.device}, dtype: {output_frame.dtype}")
                
            # Convert tensor to numpy for display
            display_frame = process_frame_for_display(output_frame)
            
            # Update display
            if not args.headless:
                pygame_surface = pygame.surfarray.make_surface(display_frame)
                window.blit(pygame_surface, (0, 0))
                pygame.display.flip()
            
            # Save frame if in headless mode or save_frames is enabled
            if args.headless or args.save_frames:
                # Create output directory if it doesn't exist
                os.makedirs(args.output_dir, exist_ok=True)
                # Save frame as image
                frame_pil = Image.fromarray(display_frame)
                frame_pil.save(os.path.join(args.output_dir, f"frame_{frame_count:04d}.png"))
                frame_count += 1
            
            # Update frames for next iteration
            frames.append(output_frame.detach().cpu())
            frames = frames[1:]
            # Make sure all frames are on the same device before stacking
            frames_tensor = torch.stack([frame.to(args.device) for frame in frames])
            
            # Cap the framerate
            if not args.headless:
                clock.tick(FPS)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if not args.headless:
            pygame.quit()

def main():
    # Parse arguments
    args = pyrallis.parse(InferenceArgs)
    
    # Print SSH-related settings
    if args.headless:
        print(f"Running in headless mode. Frames will be saved to {args.output_dir}")
    elif args.x11_display:
        print("Using X11 display for SSH forwarding. Make sure you're using 'ssh -X' or 'ssh -Y'")
    
    # Load the config file using the utility function
    print(f"Loading config from {args.config}")
    config = read_config(args.config)
    
    # Run inference
    run_pacman_inference(config, args)

if __name__ == "__main__":
    main()
