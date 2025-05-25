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
        # Encode frames
        frames = []
        for b in sequence:
            img = b['frame_image']
            if self.vae is not None and self.load_vae_feat:
                x = self.transform(img).unsqueeze(0).to(next(self.vae.parameters()).device)
                z = self.vae.encoder(x).cpu().squeeze(0)
                frames.append(z)
            else:
                frames.append(self.transform(img))
        frames = torch.stack(frames)  # [L, C, H, W]
        # Encode actions
        actions = [self._cached_one_hot[b['action']] for b in sequence]
        actions = torch.stack(actions).unsqueeze(0)  # [1, L, 5]
        # Stagger offset logic
        stagger_offset = 1
        L = self.sequence_length
        num_preds = max(L - stagger_offset - 1, 0)
        # Observations
        if num_preds > 0:
            obs_seq = frames[stagger_offset : stagger_offset + L]  # [num_preds, C, H, W]
            obs = obs_seq.reshape(-1, frames.shape[2], frames.shape[3])  # [(num_preds*C), H, W]
        else:
            obs = torch.empty((0, frames.shape[2], frames.shape[3]))
        # Target image
        img = frames[-1]  # [C, H, W]
        # Target actions and mask
        y = actions[:, :L, :]  # [1, L, 5]
        y_mask = torch.ones((1, L), dtype=torch.float32)
        # Data info
        data_info = {
            'episode': sequence[-1].get('episode', 0),
            'done': sequence[-1].get('done', False),
        }
        return {
            'obs': obs,
            'img': img,
            'y': y,
            'y_mask': y_mask,
            'data_info': data_info,
        }

    def __len__(self):
        # Number of examples (approx) from the HF dataset
        return self.dataset.info.splits['train'].num_examples
