import torch
from torch.utils.data import IterableDataset
from collections import deque
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from diffusion.data.builder import DATASETS
from diffusion.utils.logger import get_root_logger

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
        **kwargs,
    ):
        # Image transform pipeline
        self.logger = get_root_logger()
        self.transform = transform or transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Lambda(make_square),
            transforms.Resize(resolution),
            transforms.functional.hflip,
            transforms.Lambda(rotate_90_clockwise),
            transforms.ToTensor(),
            transforms.Lambda(to_float16),  # Convert to float16
        ])
        self.vae = vae
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

    def __iter__(self):
        """Yield processed sequences as dicts with keys: obs, img, y, y_mask, data_info."""
        for sample in self.dataset:
            self._buffer.append(sample)
            if len(self._buffer) == self.sequence_length:
                yield self._process_sequence(list(self._buffer))

    def _process_sequence(self, sequence):
        # L_config is self.sequence_length from the YAML (expected to be 6)
        L_config = self.sequence_length

        if L_config < 2:
            self.logger.error(
                f"Dataset sequence_length ({L_config}) in your YAML must be at least 2."
            )
            # Fallback for invalid configuration
            C_channels = 4 # Assuming VAE output channels
            H_res, W_res = (self.resolution, self.resolution)
            action_dim = 5
            obs = torch.empty(((L_config -1 if L_config > 0 else 0) * C_channels, H_res, W_res))
            img = torch.empty((C_channels, H_res, W_res))
            y = torch.empty((1, (L_config -1 if L_config > 0 else 0), action_dim))
            y_mask = torch.empty((1, (L_config -1 if L_config > 0 else 0)))
            data_info = {'episode': 0, 'done': False}
            return {'obs': obs, 'img': img, 'y': y, 'y_mask': y_mask, 'data_info': data_info}

        # Encode frames
        frames_list = []
        for b_idx, b in enumerate(sequence): # sequence here is the deque buffer of length L_config
            pil_img = b['frame_image']
            if self.vae is not None and self.load_vae_feat:
                # Assuming self.transform prepares for VAE and VAE outputs [C_vae, H, W]
                # C_vae is likely 4 based on previous error analysis.
                x = self.transform(pil_img).unsqueeze(0).to(next(self.vae.parameters()).device)
                z = self.vae.encoder(x).cpu().squeeze(0) # Shape: [C_channels, H, W]
                frames_list.append(z)
            else:
                # If not using VAE, transform should give [C_raw, H, W]
                frames_list.append(self.transform(pil_img))
        
        # frames_tensor shape: [L_config, C_channels, H, W]
        frames_tensor = torch.stack(frames_list)
        C_channels = frames_tensor.shape[1] # Get actual channels from data (e.g., 4 for VAE)
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: L_config (self.sequence_length) = {L_config}")
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: frames_tensor original shape = {frames_tensor.shape}")
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Detected C_channels = {C_channels}")

        # Encode actions
        # actions_list will have L_config actions
        actions_list = [self._cached_one_hot[b['action']] for b in sequence]
        # actions_tensor_orig shape: [L_config, 5_action_dim]
        actions_tensor_orig = torch.stack(actions_list)
        # actions_tensor shape: [1, L_config, 5_action_dim]
        actions_tensor = actions_tensor_orig.unsqueeze(0)

        # Number of frames for observation and corresponding actions
        N_obs_y_frames = L_config - 1 # This should be 5 if L_config is 6
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Calculated N_obs_y_frames = {N_obs_y_frames}")

        # Observations ('obs'): First N_obs_y_frames (i.e., L_config-1 frames)
        # obs_seq shape: [N_obs_y_frames, C_channels, H, W]
        obs_seq = frames_tensor[0:N_obs_y_frames, :, :, :]
        # obs_flat shape: [(N_obs_y_frames * C_channels), H, W]
        obs_flat = obs_seq.reshape(-1, frames_tensor.shape[2], frames_tensor.shape[3])
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: obs_flat shape (before noise) = {obs_flat.shape}")
        
        # Add noise to observation frames
        # Noise is added to the already selected N_obs_y_frames
        noise = torch.randn_like(obs_flat)
        # Using a fixed or mildly random scale for noise as in original.
        # Adjust 0.1 if you need more/less noise.
        noisy_obs = obs_flat + 0.4 * torch.rand(1).item() * noise 

        # Target Image ('img'): The last frame of the L_config sequence
        # img_target shape: [C_channels, H, W]
        img_target = frames_tensor[-1, :, :, :]

        # Target Actions ('y'): Actions from the 2nd timestep up to L_config
        # This means y has N_obs_y_frames (i.e., L_config-1) action steps.
        # y_target shape: [1, N_obs_y_frames, 5_action_dim]
        y_target = actions_tensor[:, 1:L_config, :]

        # Action Mask ('y_mask')
        # y_mask shape: [1, N_obs_y_frames]
        y_mask = torch.ones((1, N_obs_y_frames), dtype=torch.float32)
        
        data_info = {
            'episode': sequence[-1].get('episode', 0), # Get info from the last sample in the deque
            'done': sequence[-1].get('done', False),
        }
        
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning noisy_obs shape = {noisy_obs.shape}")
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning img_target shape = {img_target.shape}")
        self.logger.info(f"[PacmanDatasetSimple DEBUG] _process_sequence: Returning y_target shape = {y_target.shape}")

        return {
            'obs': noisy_obs,         # Tensor, shape: [(L_config-1)*C, H, W]
            'img': img_target,        # Tensor, shape: [C, H, W]
            'y': y_target,            # Tensor, shape: [1, L_config-1, 5]
            'y_mask': y_mask,         # Tensor, shape: [1, L_config-1]
            'data_info': data_info,   # dict: {episode: int, done: bool}
        }

    def __len__(self):
        # Number of examples (approx) from the HF dataset
        return self.dataset.info.splits['train'].num_examples
