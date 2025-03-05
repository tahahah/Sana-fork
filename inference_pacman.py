# Inference script for Pacman model
import os
import os.path as osp
import time
import numpy as np
import torch
from torch.serialization import validate_hpu_device
import pygame
from PIL import Image
import torchvision.transforms as transforms
from diffusion import DPMS
from diffusion.model.builder import get_vae, vae_encode, vae_decode, build_model
from diffusion.utils.config import SanaConfig
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.data.datasets.pacman_data import convert_to_rgb, make_square, rotate_90_clockwise, to_float16
from diffusion.data.builder import build_dataset
import pyrallis
from pathlib import Path
from dataclasses import dataclass, asdict
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
    
    # Load a sample from the dataset for initial frames
    print("Loading initial frames from dataset...")
    try:
        # Build the dataset
        image_size = config.model.image_size
        val_dataset = build_dataset(
            config.data, 
            resolution=image_size, 
            aspect_ratio_type=config.model.aspect_ratio_type, 
            vae_downsample_rate=config.vae.vae_downsample_rate, 
            vae=vae # We want images to come as raw RGB from dataset, we will encode them after
        )
        
        # Get a sample from the dataset
        dataset_iter = iter(val_dataset)
        for _ in range(4):  # Skip the first 4 iterations
            next(dataset_iter)
        sample = next(dataset_iter)  # Get the 5th iteration

        # Extract frames from the sample
        if 'img' in sample and 'obs' in sample:
            # Get the input frame (img) and observation frames (obs)
            initial_img = sample['img'].to(args.device, dtype=dtype)
            initial_obs = sample['obs'].to(args.device, dtype=dtype)
            print(f"Loaded initial frame from dataset with shape: {initial_img.shape}")
            print(f"Loaded initial observations with shape: {initial_obs.shape}")
            
            # Also get actions if available
            if 'y' in sample:
                initial_actions = sample['y'].to(args.device, dtype=dtype)
                print(f"Loaded initial actions with shape: {initial_actions.shape}")
            else:
                initial_actions = None
        else:
            # Fall back to blank frame if dataset doesn't have expected format
            print("Dataset sample doesn't contain expected fields, using blank frames")
            initial_img = torch.zeros((3, resolution, resolution), dtype=dtype, device=args.device)
            initial_obs = torch.zeros((3 * (seq_len - 1), resolution, resolution), dtype=dtype, device=args.device)
            initial_actions = None
    except Exception as e:
        print(f"Error loading from dataset: {e}")
        print("Using blank frames instead")
        initial_img = torch.zeros((3, resolution, resolution), dtype=dtype, device=args.device)
        initial_obs = torch.zeros((3 * (seq_len - 1), resolution, resolution), dtype=dtype, device=args.device)
        initial_actions = None
    
    # Set up the current input frame
    current_img = initial_img.clone()
    
    # Set up the observation history - should be in format [(seq_len-1)*C, H, W]
    current_obs = initial_obs.clone()
    
    # # Set up actions (initially all NO_ACTION or from dataset if available)
    # if initial_actions is not None and initial_actions.shape[1] == seq_len - 1:
    #     # Use actions from dataset
    #     actions = [one_hot_encode(initial_actions[0, i].argmax().item(), dtype=dtype).to(args.device) 
    #               for i in range(initial_actions.shape[1])]
    # else:
    #     # Default to NO_ACTION
    #     actions = [one_hot_encode(4, dtype=dtype).to(args.device) for _ in range(seq_len)]
    
    actions = initial_actions.clone()

    # Main loop
    running = True
    current_action = 4  # Start with NO_ACTION
    frame_count = 0
    
    # Create output directory if in headless mode
    if args.headless or args.save_frames:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving frames to {args.output_dir}")
    
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
                        print(f"Action: {current_action}")
                elif event.type == pygame.KEYUP:
                    if event.key in ACTION_MAP:
                        current_action = 4  # Reset to NO_ACTION when key is released
        
        # Update action sequence
        actions.append(one_hot_encode(current_action, dtype=dtype).to(args.device))
        actions = actions[1:]  # Remove oldest action
        
        if args.debug:
            print(f"Current action: {current_action}")
            print(f"Actions sequence: {[a.argmax().item() for a in actions]}")
            
        # Prepare inputs for model
        # 1. Add batch dimension to obs tensor (format is already [(seq_len-1)*C, H, W])
        obs_tensor = current_obs.unsqueeze(0)  # Add batch dimension [1, (seq_len-1)*C, H, W]

        # Debug prints
        if args.debug:
            print(f"Observation tensor shape: {obs_tensor.shape}")
        
        # 2. Get the current input frame [C, H, W]
        img_tensor = current_img.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        
        # Debug prints
        if args.debug:
            print(f"Input image tensor shape: {img_tensor.shape}")
        
        # 3. Generate initial noise like the input frame
        # This matches run_sampling which uses torch.randn_like(img)
        z = torch.randn_like(img_tensor)
        
        if args.debug:
            print(f"Z shape: {z.shape}")
        
        # 4. Prepare action tensor: [B, 1, S, A]
        action_tensor = torch.stack(actions).unsqueeze(0).unsqueeze(1)  # [S, A] -> [1, 1, S, A]
        action_tensor = action_tensor.to(model_device)
        
        if args.debug:
            print(f"Action tensor shape: {action_tensor.shape}")
            
        # 5. Create null action tensor for classifier-free guidance
        null_action = torch.zeros_like(action_tensor)
        null_action[:, :, :, -1] = 1.0  # Set last dimension (NO_ACTION) to 1.0
        
        # 6. Prepare model kwargs - pass observations directly as in run_sampling
        hw = [img_tensor.shape[-2], img_tensor.shape[-1]]
        ar = hw[0] / hw[1]
        model_kwargs = {
            "data_info": {"img_hw": hw, "aspect_ratio": ar},
            "mask": None,
            "obs": obs_tensor.to(model_device),  # Pass raw observations directly to model
        }
        
        # 7. Set up DPM solver and run directly on z (no pre-encoding with VAE needed)
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=action_tensor,
            uncondition=null_action,
            cfg_scale=4.5,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        
        # 8. Run the model - note: in run_sampling, the model is run on raw z, not encoded z
        with torch.no_grad():
            # Run the diffusion model directly on z (noise tensor)
            # Make sure z is on the right device
            z = z.to(model_device)
            
            # Run the sampling
            denoised = dpm_solver.sample(
                z,
                steps=40,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=config.scheduler.flow_shift,
            )
            
            # Decode the output with the VAE (only for display)
            display_output = denoised.clone()
            display_output = display_output.to(vae_device, dtype=torch.float16)
            decoded_frame = vae_decode(config.vae.vae_type, vae, display_output)
            
            if args.debug:
                print(f"Output frame shape: {decoded_frame.shape}")
        
        # Display the frame
        if not args.headless:
            # Convert tensor to numpy for display
            display_frame = process_frame_for_display(decoded_frame[0])
            
            # Display the frame
            pygame.surfarray.blit_array(window, display_frame)
            pygame.display.flip()
        
        # Save frame if in headless mode or save_frames is enabled
        if args.headless or args.save_frames:
            # Convert tensor to PIL image and save
            frame_np = decoded_frame[0].permute(1, 2, 0).detach().cpu().numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
            frame_pil = frame_pil.rotate(-90, expand=True)
            frame_pil = frame_pil.transpose(Image.FLIP_LEFT_RIGHT)
            frame_pil.save(osp.join(args.output_dir, f"frame_{frame_count:04d}.png"))
            frame_count += 1
        
        # Update for next iteration
        # 1. Update the input frame with the raw denoised output (not decoded)
        current_img = denoised[0].detach().cpu()
        
        # 2. Update the observation frames by shifting
        # The obs format is [(seq_len-1)*C, H, W]
        C = 3  # RGB channels
        channel_per_frame = C  # Each frame has C channels
        total_channels = current_obs.shape[0]  # Total channels in observations
        
        if total_channels > channel_per_frame:
            # Remove the oldest frame's channels
            new_obs_start = current_obs[channel_per_frame:].clone()
            # Add the current input frame channels
            new_obs = torch.cat([new_obs_start, current_img], dim=0)
        else:
            # If we only have one frame in history, just use the current frame
            new_obs = current_img.clone()
        
        # Update the observation tensor
        current_obs = new_obs
        
        # Cap the framerate
        if not args.headless:
            clock.tick(FPS)
    
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
