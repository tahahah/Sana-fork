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
import traceback
import logging
from typing import Optional

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

def convert_to_rgb(img):
    return img.convert("RGB")

def rotate_90_clockwise(img):
    return img.rotate(90, expand=True)

def to_float16(x):
    """Convert tensor to float16."""
    return x.half()

class PacmanIterator:
    """Iterator class for PacmanDataset"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset._init_worker_state()
        
        # Start processing thread if not already running
        if self.dataset._worker_thread is None or not self.dataset._worker_thread.is_alive():
            self.dataset._stop_event.clear()
            self.dataset._worker_thread = threading.Thread(target=self.dataset._process_sequences, daemon=True)
            self.dataset._worker_thread.start()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            # Try to get an item with timeout to avoid hanging
            item = self.dataset.processed_sequences.get(timeout=1.0)
            if item is None:  # Sentinel value
                raise StopIteration
            return item
        except queue.Empty:
            if not self.dataset._worker_thread.is_alive():
                raise StopIteration
            # If thread is still alive, try again
            return self.__next__()
        except Exception as e:
            self.dataset.logger.error(f"Error in iterator: {e}")
            self.dataset.stop_worker()
            raise StopIteration

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
        vae=None,  
        **kwargs,
    ):
        self.logger = get_root_logger() # if config is None else get_root_logger(osp.join(config.work_dir, "train_log.log"))
        self.transform = transform if not load_vae_feat else None
        self.load_vae_feat = load_vae_feat
        self.load_text_feat = load_text_feat
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.vae = vae
        self.config = config
        # Default to fp32 if no config provided
        self.mixed_precision = "fp32"
        if config is not None and hasattr(config, 'model') and hasattr(config.model, 'mixed_precision'):
            self.mixed_precision = config.model.mixed_precision
        
        # Create blank image for padding
        self.blank_image = Image.new('RGB', (512, 512), 'black')
        self._cached_blank_frame = None
        self._cached_blank_latent = None  
        
        # Cache for encoded frames
        self._encoded_frames_cache = {}  # Maps frame hash to encoded frame
        self._cache_size = sequence_length * 4  # Keep cache size reasonable
        
        # Load streaming dataset
        self.dataset = load_dataset(
            "Tahahah/PacmanDataset_3", 
            split="train", 
            verification_mode="no_checks", 
            streaming=True
        )

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Lambda(convert_to_rgb),
                transforms.Lambda(make_square),  # Make the image square with padding
                transforms.Resize(512),          # Resize to 512x512
                transforms.functional.hflip,     # Horizontal mirror flip
                transforms.Lambda(rotate_90_clockwise),  # Rotate 90 degrees clockwise
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(to_float16),  # Convert to float16
            ])
        
        # Cache one-hot vectors in float16
        self._cached_one_hot = {
            i: torch.zeros(5, dtype=torch.float16).scatter_(0, torch.tensor(i), 1)
            for i in range(5)
        }
        
        self.logger.info(f"Initialized Pacman streaming dataset with buffer size {buffer_size}")
        
        # These will be initialized in worker processes
        self.sample_buffer = None
        self.sequence_buffer = None
        self.processed_sequences = None
        self._stop_event = None
        self._worker_thread = None
        
    def _init_worker_state(self):
        """Initialize worker-specific state (called in each worker process)"""
        if self.sample_buffer is None:
            self.sample_buffer = deque(maxlen=self.buffer_size)
            self.sequence_buffer = deque(maxlen=self.sequence_length)
            self.processed_sequences = queue.Queue(maxsize=self.prefetch_factor * self.buffer_size)
            self._stop_event = threading.Event()
            
    def __iter__(self):
        """Return an iterator over the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        return PacmanIterator(self)

    def __del__(self):
        """Cleanup when the dataset is destroyed."""
        try:
            self.stop_worker()
        except:
            pass  # Ignore errors during cleanup

    def stop_worker(self):
        """Stop the prefetch worker thread."""
        if hasattr(self, '_worker_thread') and self._worker_thread is not None:
            if hasattr(self, '_stop_event'):
                self._stop_event.set()
            if self._worker_thread.is_alive():
                try:
                    self._worker_thread.join(timeout=0.1)  # Reduced timeout
                except:
                    pass  # Ignore join errors
            self._worker_thread = None
            
    def _process_sequence(self, sequence):
        """Process a sequence of samples into the required format."""
        if len(sequence) < self.sequence_length:
            # Pad with blanks at the start
            padding_length = self.sequence_length - len(sequence)
            if self.vae is not None and not self.load_vae_feat:
                frames = ([self.blank_latent] * padding_length +
                         [self._vae_encode_single(self.transform(b['frame_image'])) for b in sequence])
            else:
                frames = ([self.blank_frame] * padding_length +
                         [self.transform(b['frame_image']) for b in sequence])
            actions = ([self._one_hot_encode(4)] * padding_length +
                     [self._one_hot_encode(b['action']) for b in sequence])
        else:
            # Use last sequence_length samples
            if self.vae is not None and not self.load_vae_feat:
                frames = [self._vae_encode_single(self.transform(b['frame_image'])) 
                         for b in sequence[-self.sequence_length:]]
            else:
                frames = [self.transform(b['frame_image']) 
                         for b in sequence[-self.sequence_length:]]
            actions = [self._one_hot_encode(b['action']) 
                      for b in sequence[-self.sequence_length:]]

        # Stack tensors
        frames = torch.stack(frames)  # [seq_len, C, H, W] or [seq_len, 32, 16, 16] if VAE encoded
        actions = torch.stack(actions)  # [seq_len, 5]
        
        B, C, H, W = frames.shape
        frames = frames.view(-1, H, W)  # Flatten sequence and channels
            
        # Reshape actions to match model's expected input shape: [1, seq_len, 5]
        actions = actions.unsqueeze(0)  # Add batch dimension
        
        return {
            'obs': frames[:-C, : , :],  # [(seq_len-1)*C, H, W]
            'img': frames[-C:, :, :],   # [C, H, W] 
            'y': actions[:, :-1, :],  # [1, seq_len-1, 5]
            'y_mask': torch.ones(1, actions.shape[1] - 1),  # [1, seq_len-1]
            'data_info': {
                'episode': sequence[-1].get('episode', 0),
                'done': sequence[-1].get('done', False)
            }
        }

    def _one_hot_encode(self, action, num_classes=5):
        """Convert action to one-hot encoding.        
            LEFT = 0
            RIGHT = 1
            UP = 2
            DOWN = 3
            NO_ACTION = 4
        """
        return self._cached_one_hot[action]
    
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
                self.logger.error(f"Error in prefetch worker: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                break
    
    def _get_frame_hash(self, frame_image):
        """Get a unique hash for a frame image for caching."""
        if isinstance(frame_image, torch.Tensor):
            # For tensors, use numpy bytes
            return hash(frame_image.cpu().numpy().tobytes())
        else:
            # For PIL images, use image bytes
            return hash(frame_image.tobytes())

    def _vae_encode_single(self, frame):
        """Encode a single frame, using cache if available."""
        if self.vae is None or self.load_vae_feat:
            return frame
            
        frame_hash = self._get_frame_hash(frame)
        if frame_hash in self._encoded_frames_cache:
            return self._encoded_frames_cache[frame_hash]
            
        try:
            device = next(self.vae.parameters()).device
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=(self.mixed_precision == "fp16" or self.mixed_precision == "bf16"),
                ):
                    frame = frame.unsqueeze(0).to(device)  # Add batch dimension
                    latent = self.vae.encode(frame).cpu()  # Direct encoding without sampling
                    
                    # Cache the result
                    self._encoded_frames_cache[frame_hash] = latent.squeeze(0)
                    
                    # Maintain cache size
                    if len(self._encoded_frames_cache) > self._cache_size:
                        # Remove oldest items
                        oldest_key = next(iter(self._encoded_frames_cache))
                        del self._encoded_frames_cache[oldest_key]
                    
                    return latent.squeeze(0)
        except Exception as e:
            self.logger.error(f"Error in VAE encoding: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise

    @property
    def blank_frame(self):
        """Cached transformed blank frame."""
        if self._cached_blank_frame is None:
            self._cached_blank_frame = self.transform(self.blank_image)
        return self._cached_blank_frame

    @property
    def blank_latent(self):
        """Cached VAE-encoded blank frame."""
        if self._cached_blank_latent is None and self.vae is not None:
            with torch.no_grad():
                frame = self.blank_frame.unsqueeze(0).to(next(self.vae.parameters()).device)
                self._cached_blank_latent = self.vae.encode(frame).cpu().squeeze(0)
        return self._cached_blank_latent

    def _process_sequences(self):
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
                self.logger.error(f"Error in prefetch worker: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                break
    
    def __len__(self):
        return self.dataset.info.splits['train'].num_examples

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
