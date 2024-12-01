import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import os.path as osp
import numpy as np
from datasets import load_dataset
from collections import deque
from itertools import islice

from diffusion.data.builder import DATASETS
from diffusion.utils.logger import get_root_logger

@DATASETS.register_module()
class PacmanDataset(IterableDataset):
    def __init__(
        self,
        data_dir="",  # Not used, kept for compatibility
        transform=None,
        resolution=512,
        load_vae_feat=False,
        load_text_feat=False,
        sequence_length=64,
        buffer_size=1000,  # Size of the sample buffer for batching
        config=None,
        **kwargs,
    ):
        self.logger = get_root_logger() if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        
        # Create blank image for padding
        self.blank_image = Image.new('RGB', (552, 456), 'black')
        
        # Load streaming dataset
        self.dataset = load_dataset(
            "Tahahah/PacmanDataset_3", 
            split="train", 
            verification_mode="no_checks", 
            streaming=True
        )
        
        # Initialize buffers
        self.sample_buffer = deque(maxlen=buffer_size)
        self.sequence_buffer = []
        
        self.logger.info(f"Initialized Pacman streaming dataset with buffer size {buffer_size}")
    
    def _process_sequence(self, sequence):
        """Process a sequence of samples into the required format."""
        if len(sequence) < self.sequence_length:
            # Pad with blanks
            frames = ([self.transform(self.blank_image)] * (self.sequence_length - len(sequence)) +
                     [self.transform(Image.fromarray(b['frame'])) for b in sequence])
            actions = ([0] * (self.sequence_length - len(sequence)) +
                     [b['action'] for b in sequence])
        else:
            # Use full sequence
            frames = [self.transform(Image.fromarray(b['frame'])) for b in sequence[-self.sequence_length:]]
            actions = [b['action'] for b in sequence[-self.sequence_length:]]

        # Stack tensors
        frames = torch.stack(frames)  # [seq_len, C, H, W]
        actions = torch.tensor(actions, dtype=torch.long)  # [seq_len]
        
        return {
            'img': frames[-1],  # Latest frame
            'y': actions.unsqueeze(0),  # [1, seq_len]
            'y_mask': torch.ones(1, 1, actions.shape[0]),  # [1, 1, seq_len]
            'data_info': {
                'frames': frames,
                'episode': sequence[-1].get('episode', 0),
                'done': sequence[-1].get('done', False)
            }
        }

    def _fill_buffer(self):
        """Fill the sample buffer with new samples."""
        while len(self.sample_buffer) < self.buffer_size:
            try:
                sample = next(iter(self.dataset))
                self.sample_buffer.append(sample)
            except StopIteration:
                break

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        def sample_generator():
            while True:
                # Fill buffer if needed
                self._fill_buffer()
                if not self.sample_buffer:
                    break
                
                # Get a batch of samples
                batch_size = min(len(self.sample_buffer), self.buffer_size)
                samples = [self.sample_buffer.popleft() for _ in range(batch_size)]
                
                # Process each sample
                for sample in samples:
                    self.sequence_buffer.append(sample)
                    if len(self.sequence_buffer) > self.sequence_length:
                        self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]
                    
                    if len(self.sequence_buffer) > 0:
                        yield self._process_sequence(self.sequence_buffer)
        
        # If using multiple workers, split the iterator
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            it = sample_generator()
            for i, sample in enumerate(it):
                if i % num_workers == worker_id:
                    yield sample
        else:
            yield from sample_generator()


@DATASETS.register_module()
class PacmanDatasetMS(PacmanDataset):
    def __init__(self, aspect_ratio_type="ASPECT_RATIO_1024", **kwargs):
        super().__init__(**kwargs)
        # Add multi-scale specific initialization
        self.base_size = int(aspect_ratio_type.split("_")[-1])
        self.aspect_ratio = eval(aspect_ratio_type)
        self.interpolate_mode = InterpolationMode.BICUBIC

    def __iter__(self):
        for data in super().__iter__():
            # Add any multi-scale specific processing here if needed
            yield data
