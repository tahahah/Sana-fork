import torch
from torch.utils.data import IterableDataset
from collections import deque
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from diffusion.data.builder import DATASETS
from diffusion.utils.logger import get_root_logger

# For debugging, add a handler that prints to stderr immediately if logs don't appear
import logging
import sys

# Utility functions for image preprocessing

def make_square(image):
    """Pad image to square by adding black borders."""
    width, height = image.size
    max_dim = max(width, height)
    padding = [
        (max_dim - width) // 2,
        (max_dim - height) // 2,
        (max_dim - width + 1) // 2,
        (max_dim - height + 1) // 2,
    ]
    return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')


def convert_to_rgb(img):
    return img.convert("RGB")


def rotate_90_clockwise(img):
    return img.rotate(90, expand=True)

def to_float16(x):
    """Convert tensor to float16."""
    return x.half()

@DATASETS.register_module()
class PacmanDatasetSimple(IterableDataset):
    """
    Simplified Pacman dataset without padding or background buffering.
    Drop-in replacement for pacman_data_copy.PacmanDataset.
    """
    def __init__(
        self,
        data_dir="",
        transform=None,
        resolution=512,
        load_vae_feat=False,
        load_text_feat=False,
        sequence_length=64,
        buffer_size=None,
        prefetch_factor=None,
        config=None,
        vae=None,
        debug=False,
        is_validation_run=False, # New parameter
        **kwargs,
    ):
        # Image transform pipeline
        self.logger = get_root_logger()
        # Ensure logger is at least INFO level for these messages
        self.logger.setLevel("INFO") 
        if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in self.logger.handlers):
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
        if transform is None: # Check if a transform was passed in
            self.logger.info("[PacmanDatasetSimple DEBUG] __init__: Using default transforms.")
            self.transform = transforms.Compose([
                transforms.Lambda(convert_to_rgb),
                transforms.Lambda(make_square),
            transforms.Resize(resolution),
            transforms.functional.hflip,
            transforms.Lambda(rotate_90_clockwise),
                transforms.ToTensor(),
                transforms.Lambda(to_float16),  # Convert to float16
            ])
        else:
            self.logger.info(f"[PacmanDatasetSimple DEBUG] __init__: Using provided transform: {type(transform)}")
            self.transform = transform # Use the provided transform

        self.vae = vae
        
        self.config = config
        # Default to fp32 if no config provided
        self.mixed_precision = "fp32"
        if config is not None and hasattr(config, 'model') and hasattr(config.model, 'mixed_precision'):
            self.mixed_precision = config.model.mixed_precision

        self.load_vae_feat = load_vae_feat
        self.sequence_length = sequence_length
        # One-hot cache for actions
        self._cached_one_hot = {
            i: torch.eye(5, dtype=torch.float32)[i] for i in range(5)
        }
        # Load streaming dataset
        self.dataset = load_dataset(
            "Tahahah/PacmanDataset_3",
            split="train",
            verification_mode="no_checks",
            streaming=True,
        )
        # Sliding window buffer
        self._buffer = deque(maxlen=self.sequence_length)
        self.debug = debug # Store debug flag
        self.is_validation_run = is_validation_run # Store validation run flag
        self.last_raw_validation_data = None # Initialize for validation logging

    def __iter__(self):
        """Yield processed sequences as dicts with keys: obs, img, y, y_mask, data_info."""
        for sample in self.dataset:
            self._buffer.append(sample)
            if len(self._buffer) == self.sequence_length:
                yield self._process_sequence(list(self._buffer))

    def _process_sequence(self, sequence): # sequence is a list of L_config items from the buffer
        L_config = self.sequence_length

        # L_config >= 2 is implicitly handled as __iter__ calls this only when buffer is full to sequence_length,
        # and __init__ has checks for sequence_length.

        if self.is_validation_run:
            if self.debug:
                self.logger.info(f"[DEBUG PacmanDatasetSimple._process_sequence] Validation run. Storing raw data. Num frames in sequence: {len(sequence)}")
            
            # Apply stagger offset to raw validation data to match model input
            N_obs_y_len = L_config - 1 # This should be consistent with the N_obs_y_len calculated later
            stagger_offset = 1 # This should be consistent with the stagger_offset calculated later

            raw_pil_images = [b['frame_image'] for b in sequence[stagger_offset : stagger_offset + N_obs_y_len]]
            raw_actions = [b['action'] for b in sequence[0 : N_obs_y_len]] # Actions are not staggered, they correspond to the obs frames
            self.last_raw_validation_data = {'raw_frames': raw_pil_images, 'raw_actions': raw_actions}

        frames_list = []
        val_obs = []
        for b_idx, b_data in enumerate(sequence):
            pil_img = b_data['frame_image']
            if self.vae is not None and not self.load_vae_feat:
                if self.debug: print(f"[STDERR DEBUG] PacmanDatasetSimple _process_sequence: Using VAE for frame {b_idx}", file=sys.stderr)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=(self.mixed_precision in ["fp16", "bf16"])):
                    x = self.transform(pil_img).unsqueeze(0).to(next(self.vae.parameters()).device)
                    frames_list.append(self.vae.encoder(x).cpu().squeeze(0))
            else:
                if self.debug: print(f"[STDERR DEBUG] PacmanDatasetSimple _process_sequence: Not using VAE for frame {b_idx}", file=sys.stderr)
                frames_list.append(self.transform(pil_img))
            if self.is_validation_run:
                val_obs.append(pil_img)
        
        frames_tensor = torch.stack(frames_list) # Shape: [L_config, C, H, W]
        C_channels = frames_tensor.shape[1]
        
        actions_list = [self._cached_one_hot[b_data['action']] for b_data in sequence]
        actions_tensor = torch.stack(actions_list).unsqueeze(0) # Shape: [1, L_config, ActionDim]

        # N_obs_y_len is the number of frames in the observation sequence and number of target actions.
        N_obs_y_len = L_config - 1 
        stagger_offset = 1 # Start observations from frames_tensor[stagger_offset]

        if self.debug:
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: L_config={L_config}, N_obs_y_len={N_obs_y_len}, stagger_offset={stagger_offset}")
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: frames_tensor shape: {frames_tensor.shape}, C_channels: {C_channels}")
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: actions_tensor shape: {actions_tensor.shape}")

        # Observations ('obs'): N_obs_y_len frames, starting from stagger_offset
        # e.g., if L_config=6, N_obs_y_len=5, stagger=1: obs uses F_1, F_2, F_3, F_4, F_5
        obs_seq = frames_tensor[stagger_offset : stagger_offset + N_obs_y_len, :, :, :]
        val_obs = val_obs[stagger_offset : stagger_offset + N_obs_y_len]
        
        # obs_flat shape: [N_obs_y_len * C_channels, H, W]
        obs_flat = obs_seq.reshape(-1, frames_tensor.shape[2], frames_tensor.shape[3])
        if self.debug: self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: obs_flat shape (before noise) = {obs_flat.shape}")
        
        # Add noise to observations. N_obs_y_len >= 1 since L_config >= 2.
        noisy_obs = obs_flat + 0.5 * torch.rand(1).item() * torch.randn_like(obs_flat)

        # Target Image ('img'): Last frame of the L_config window (F_{L_config-1})
        img_target = frames_tensor[-1, :, :, :] 
        # Target Actions ('y'): N_obs_y_len actions, A_0 to A_{N_obs_y_len - 1}
        # e.g., if L_config=6, N_obs_y_len=5: y uses A_0, A_1, A_2, A_3, A_4
        y_target = actions_tensor[:, 0 : N_obs_y_len, :] # Shape: [1, N_obs_y_len, ActionDim]

        # Mask for target actions ('y_mask')
        y_mask = torch.ones((1, N_obs_y_len), dtype=torch.float32, device=frames_tensor.device)

        # Data info from the last sample in the original L_config sequence segment
        data_info = {
            'episode': sequence[-1].get('episode', 0), 
            'done': sequence[-1].get('done', False),
        }
        
        if self.debug:
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning noisy_obs shape = {noisy_obs.shape}")
            print(f"[STDERR DEBUG] PacmanDatasetSimple _process_sequence: noisy_obs shape = {noisy_obs.shape}", file=sys.stderr)
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning img_target shape = {img_target.shape}")
            self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning y_target shape = {y_target.shape}")

        return {
            'val_obs': val_obs,
            'obs': noisy_obs,         # Expected shape: [(L_config-1)*C, H, W]
            'img': img_target,        # Expected shape: [C, H, W]
            'y': y_target,            # Expected shape: [1, L_config-1, ActionDim]
            'y_mask': y_mask,         # Expected shape: [1, L_config-1]
            'data_info': data_info,
        }   

    def get_last_raw_validation_data(self):
        """Retrieves the last stored raw data for validation logging and clears it."""
        if self.debug:
            self.logger.info(f"[DEBUG PacmanDatasetSimple.get_last_raw_validation_data] Called. Data is {'not None' if self.last_raw_validation_data else 'None'}. Clearing it.")
        data = self.last_raw_validation_data
        self.last_raw_validation_data = None  # Clear after retrieval
        return data

    def __len__(self):
        # Number of examples (approx) from the HF dataset
        return self.dataset.info.splits['train'].num_examples
