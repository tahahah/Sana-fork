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
import threading
import queue

from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
from diffusion.data.builder import DATASETS
from diffusion.utils.logger import get_root_logger

def make_square(image):
    # Calculate the necessary padding to make the image square
    width, height = image.size
    max_dim = max(width, height)
    padding = [
        (max_dim - width) // 2,  # Left padding
        (max_dim - height) // 2, # Top padding
        (max_dim - width + 1) // 2,  # Right padding
        (max_dim - height + 1) // 2  # Bottom padding
    ]
    return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

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
        prefetch_factor=2,  # Number of batches to prefetch
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
        self.prefetch_factor = prefetch_factor
        
        # Create blank image for padding
        self.blank_image = Image.new('RGB', (552, 456), 'black')
        self._cached_blank_frame = None
        
        # Load streaming dataset
        self.dataset = load_dataset(
            "Tahahah/PacmanDataset_3", 
            split="train", 
            verification_mode="no_checks", 
            streaming=True
        )

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Lambda(make_square),  # Make the image square with padding
                transforms.Resize(512),          # Resize to 512x512
                transforms.functional.hflip,     # Horizontal mirror flip
                transforms.Lambda(lambda img: img.rotate(90, expand=True)),  # Rotate 90 degrees clockwise
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        
        # Initialize buffers
        self.sample_buffer = deque(maxlen=buffer_size)
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.processed_sequences = queue.Queue(maxsize=prefetch_factor * buffer_size)
        
        # Threading control
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Cache one-hot vectors
        self._cached_one_hot = {
            i: torch.zeros(5).scatter_(0, torch.tensor(i), 1)
            for i in range(5)
        }
        
        self.logger.info(f"Initialized Pacman streaming dataset with buffer size {buffer_size}")
    
    def __del__(self):
        """Cleanup when the dataset is destroyed."""
        self.stop_worker()
    
    def stop_worker(self):
        """Stop the prefetch worker thread."""
        if self._worker_thread is not None:
            self._stop_event.set()
            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
    
    @property
    def blank_frame(self):
        """Cached transformed blank frame."""
        if self._cached_blank_frame is None:
            self._cached_blank_frame = self.transform(self.blank_image)
        return self._cached_blank_frame
    
    def _one_hot_encode(self, action, num_classes=5):
        """Convert action to one-hot encoding.        
            LEFT = 0
            RIGHT = 1
            UP = 2
            DOWN = 3
            NO_ACTION = 4
        """
        return self._cached_one_hot[action]
    
    def _process_sequence(self, sequence):
        """Process a sequence of samples into the required format."""
        if len(sequence) < self.sequence_length:
            # Pad with blanks at the start
            padding_length = self.sequence_length - len(sequence)
            frames = ([self.blank_frame] * padding_length +
                     [self.transform(b['frame_image']) for b in sequence])
            actions = ([self._one_hot_encode(4)] * padding_length +
                     [self._one_hot_encode(b['action']) for b in sequence])
        else:
            # Use last sequence_length samples
            frames = [self.transform(b['frame_image']) for b in sequence[-self.sequence_length:]]
            actions = [self._one_hot_encode(b['action']) for b in sequence[-self.sequence_length:]]

        # Stack tensors - DataLoader will handle batch dimension
        frames = torch.stack(frames)  # [seq_len, C, H, W]
        actions = torch.stack(actions)  # [seq_len, 5]
        
        return {
            'img': frames[-1],  # [C, H, W] Latest frame
            'y': actions,  # [seq_len, 5]
            'y_mask': torch.ones(1, actions.shape[0]),  # [1, seq_len]
            'data_info': {
                'frames': frames,  # [seq_len, C, H, W]
                'episode': sequence[-1].get('episode', 0),
                'done': sequence[-1].get('done', False)
            }
        }

    def _fill_buffer(self):
        """Fill the sample buffer with new samples efficiently."""
        # Fill buffer in batches
        batch_size = self.buffer_size - len(self.sample_buffer)
        if batch_size <= 0:
            return
            
        try:
            # Get multiple samples at once
            samples = list(islice(iter(self.dataset), batch_size))
            self.sample_buffer.extend(samples)
        except StopIteration:
            pass

    def _prefetch_worker(self):
        """Background worker to prefetch and process sequences."""
        while not self._stop_event.is_set():
            try:
                # Fill buffer if needed
                self._fill_buffer()
                if not self.sample_buffer:
                    break
                
                # Get a batch of samples
                batch_size = min(len(self.sample_buffer), self.buffer_size)
                samples = [self.sample_buffer.popleft() for _ in range(batch_size)]
                
                # Process each sample
                for sample in samples:
                    if self._stop_event.is_set():
                        return
                    self.sequence_buffer.append(sample)
                    if len(self.sequence_buffer) > 0:
                        sequence = list(self.sequence_buffer)
                        processed = self._process_sequence(sequence)
                        self.processed_sequences.put(processed)
            except Exception as e:
                self.logger.error(f"Error in prefetch worker: {e}")
                break

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Stop any existing worker
        self.stop_worker()
        
        # Reset stop event and start new worker
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._worker_thread.start()
        
        def sample_generator():
            while not self._stop_event.is_set():
                try:
                    # Get processed sequence from queue
                    yield self.processed_sequences.get(timeout=1.0)
                except queue.Empty:
                    if not self._worker_thread.is_alive():
                        break
        
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
        try:
            self.base_size = int(aspect_ratio_type.split("_")[2])  # Gets '512' from 'ASPECT_RATIO_512_TEST'
        except (IndexError, ValueError):
            self.base_size = 512
            
        self.aspect_ratio = eval(aspect_ratio_type)
        self.interpolate_mode = InterpolationMode.BICUBIC

    def __iter__(self):
        for data in super().__iter__():
            yield data
