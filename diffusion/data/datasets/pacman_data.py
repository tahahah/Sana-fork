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
        self.logger = get_root_logger()
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
        self.blank_image = Image.new('RGB', (self.resolution, self.resolution), 'black')
        self._cached_blank_frame = None
        self._cached_blank_latent = None  
        
        # Cache for encoded frames
        self._encoded_frames_cache = {}  # Maps frame hash to encoded frame
        self._cache_size = sequence_length * 4  # Keep cache size reasonable
        
        # Load precomputed latents for target frames (if available)
        self._latent_data = None
        self._latent_idx = 0
        if load_vae_feat:
            latent_path = data_dir[0] if isinstance(data_dir, list) and len(data_dir) > 0 else data_dir
            if not latent_path.endswith('.pt'):
                latent_path = osp.join(latent_path, 'pacman_latents.pt')
            self.logger.info(f"Loading precomputed VAE latents from {latent_path}...")
            data = torch.load(latent_path, map_location='cpu', weights_only=False)
            self._latent_data = data['latents']  # [N, 4, 32, 32] fp16
            self._num_latents = len(self._latent_data)
            self.logger.info(f"Loaded {self._num_latents} precomputed latents")
            self._cached_blank_latent = torch.zeros(4, 32, 32, dtype=self._latent_data.dtype)
        
        # Always stream from HuggingFace for obs (pixel images needed by history encoder)
        self.dataset = load_dataset(
            "Tahahah/PacmanDataset_3", 
            split="train", 
            verification_mode="no_checks", 
            streaming=True
        )
        
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Lambda(convert_to_rgb),
                transforms.Lambda(make_square),
                transforms.Resize(self.resolution),
                transforms.functional.hflip,
                transforms.Lambda(rotate_90_clockwise),
                transforms.ToTensor(),
                transforms.Lambda(to_float16),
            ])
        
        # Cache one-hot vectors in float16
        self._cached_one_hot = {
            i: torch.zeros(5, dtype=torch.float16).scatter_(0, torch.tensor(i), 1)
            for i in range(5)
        }
        
        self.logger.info(f"Initialized Pacman dataset (buffer={buffer_size}, load_vae_feat={load_vae_feat})")
        
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
        if worker_info is not None and self.load_vae_feat:
            self.logger.warning("load_vae_feat=True with num_workers>1 may cause latent misalignment. Consider num_workers=0.")
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
        # Always transform pixel-space frames for obs (history encoder needs raw images)
        if self.vae is not None and not self.load_vae_feat:
            # VAE encode on-the-fly for obs (original path when VAE is available)
            if len(sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(sequence)
                frames = ([self.blank_latent] * padding_length +
                         [self._vae_encode_single(self.transform(b['frame_image'])) for b in sequence])
                actions = ([self._one_hot_encode(4)] * padding_length +
                         [self._one_hot_encode(b['action']) for b in sequence])
            else:
                frames = [self._vae_encode_single(self.transform(b['frame_image'])) 
                         for b in sequence[-self.sequence_length:]]
                actions = [self._one_hot_encode(b['action']) 
                          for b in sequence[-self.sequence_length:]]
        else:
            # Pixel-space transforms (for obs — history encoder processes these)
            if len(sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(sequence)
                frames = ([self.blank_frame] * padding_length +
                         [self.transform(b['frame_image']) for b in sequence])
                actions = ([self._one_hot_encode(4)] * padding_length +
                         [self._one_hot_encode(b['action']) for b in sequence])
            else:
                frames = [self.transform(b['frame_image']) 
                         for b in sequence[-self.sequence_length:]]
                actions = [self._one_hot_encode(b['action']) 
                          for b in sequence[-self.sequence_length:]]

        # Stack tensors
        frames = torch.stack(frames)  # [seq_len, C, H, W]
        actions = torch.stack(actions)  # [seq_len, 5]
        
        B, C, H, W = frames.shape
        frames = frames.view(-1, H, W)  # Flatten sequence and channels
            
        # Reshape actions to match model's expected input shape: [1, seq_len, 5]
        actions = actions.unsqueeze(0)  # Add batch dimension
        
        result = {
            'obs': frames[:-C, : , :],  # [(seq_len-1)*C, H, W]
            'img': frames[-C:, :, :],   # [C, H, W] — pixel space for obs VAE encoding
            'y': actions[:, :-1, :],  # [1, seq_len-1, 5]
            'y_mask': torch.ones(1, actions.shape[1] - 1),  # [1, seq_len-1]
            'data_info': {
                'episode': sequence[-1].get('episode', 0),
                'done': sequence[-1].get('done', False)
            }
        }
        
        # Attach precomputed latent for the target frame (always include key for consistent collation)
        if self.load_vae_feat and self._latent_data is not None:
            last_sample = sequence[-1]
            if 'frame_latent' in last_sample and last_sample['frame_latent'] is not None:
                result['img_latent'] = last_sample['frame_latent']  # [4, 32, 32]
            else:
                result['img_latent'] = self._cached_blank_latent  # fallback for padding frames
        
        return result

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
        batch_size = self.buffer_size - len(self.sample_buffer)
        if batch_size <= 0:
            return
            
        try:
            # Always stream from HuggingFace (obs needs pixel images for history encoder)
            samples = list(islice(iter(self.dataset), batch_size))
            for s in samples:
                if self.load_vae_feat and self._latent_data is not None:
                    s['frame_latent'] = self._latent_data[self._latent_idx]
                    self._latent_idx += 1
                    if self._latent_idx >= self._num_latents:
                        self._latent_idx = 0  # Loop for multi-epoch
                        # Reset HF stream to stay aligned with latent index
                        self.dataset = load_dataset(
                            "Tahahah/PacmanDataset_3",
                            split="train",
                            verification_mode="no_check",
                            streaming=True
                        )
                        break  # Stop fetching from old stream, restart on next _fill_buffer
                self.sample_buffer.append(s)
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
        if self.load_vae_feat and self._latent_data is not None:
            return self._num_latents
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

@DATASETS.register_module()
class PacmanMapDataset(torch.utils.data.Dataset):
    """Map-style dataset with random access. Supports precomputed VAE latents.
    Index i returns frame i as target and frames [i-seq_len+1, i] as obs sequence.
    """
    def __init__(self, resolution=256, sequence_length=4, load_vae_feat=False,
                 data_dir=None, buffer_size=200, prefetch_factor=4, vae=None, config=None, **kwargs):
        self.logger = get_root_logger()
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.load_vae_feat = load_vae_feat
        self.config = config

        # Load local dataset (Map-style, random access)
        local_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "datasets", "pacman_raw")
        self.logger.info(f"Loading local dataset from {local_path}...")
        from datasets import load_from_disk
        self.ds = load_from_disk(local_path)
        self.n_samples = len(self.ds)
        self.logger.info(f"Loaded {self.n_samples} frames")

        # Load precomputed latents
        self.latents = None
        if load_vae_feat:
            latent_path = data_dir[0] if isinstance(data_dir, list) else data_dir
            if not latent_path.endswith('.pt'):
                latent_path = osp.join(latent_path, 'pacman_latents.pt')
            self.logger.info(f"Loading precomputed latents from {latent_path}...")
            data = torch.load(latent_path, map_location='cpu', weights_only=False)
            self.latents = data['latents']
            self.latent_episodes = data['episodes']
            self.latent_actions = data['actions']
            self.latent_dones = data['dones']
            n_latents = len(self.latents)
            self.logger.info(f"Loaded {n_latents} precomputed latents")
            if n_latents < self.n_samples:
                self.logger.warning(f"Only {n_latents} latents for {self.n_samples} frames — using first {n_latents}")
                self.n_samples = n_latents

        # Build transform (same as streaming dataset)
        self.transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Lambda(make_square),
            transforms.Resize(self.resolution),
            transforms.functional.hflip,
            transforms.Lambda(rotate_90_clockwise),
            transforms.ToTensor(),
            transforms.Lambda(to_float16),
        ])

        # Blank frame for padding
        self.blank_frame = torch.zeros(3, self.resolution, self.resolution, dtype=torch.float16)
        self.blank_latent = torch.zeros(4, 32, 32, dtype=torch.float16)

        # Cache one-hot vectors
        self._one_hot = {
            i: torch.zeros(5, dtype=torch.float16).scatter_(0, torch.tensor(i), 1)
            for i in range(5)
        }

        self.logger.info(f"Initialized PacmanMapDataset (n={self.n_samples}, load_vae_feat={load_vae_feat})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Return a sequence ending at frame idx: obs=[idx-seq+1, idx-1], target=idx."""
        seq_len = self.sequence_length
        start = max(0, idx - seq_len + 1)
        actual_len = idx - start + 1  # includes target
        padding = seq_len - actual_len

        # Fully-latent path: use precomputed latents for BOTH obs and target.
        # No image decode and no runtime VAE — actions/metadata come from the
        # precomputed arrays, which are index-aligned with the latents.
        if self.load_vae_feat and self.latents is not None:
            obs_lat = [self.latents[j] for j in range(start, idx)]  # frames before target
            actions = [self._one_hot[int(self.latent_actions[j])] for j in range(start, idx + 1)]
            if padding > 0:
                obs_lat = [self.blank_latent] * padding + obs_lat
                actions = [self._one_hot[4]] * padding + actions

            # obs latents: [seq_len-1, 4, h, w] -> flatten channels [(seq_len-1)*4, h, w]
            obs_latent = torch.stack(obs_lat)
            L, C, h, w = obs_latent.shape
            obs_latent = obs_latent.reshape(L * C, h, w)
            actions = torch.stack(actions)  # [seq_len, 5]

            y = actions[:-1].unsqueeze(0)  # obs actions only -> [1, seq_len-1, 5]
            y_mask = torch.ones(1, y.shape[1])
            return {
                'obs_latent': obs_latent,       # [(seq_len-1)*4, h, w]
                'img_latent': self.latents[idx],  # [4, h, w] target
                'y': y,
                'y_mask': y_mask,
                'data_info': {
                    'episode': self.latent_episodes[idx],
                    'done': self.latent_dones[idx],
                },
            }

        # ---- pixel path (load_vae_feat=False): trainable pixel-space obs ----
        # Get frames
        frames = []
        actions = []
        for j in range(start, idx + 1):
            sample = self.ds[j]
            frames.append(self.transform(sample['frame_image']))
            actions.append(self._one_hot[sample['action']])

        # Pad at the beginning if needed
        if padding > 0:
            frames = [self.blank_frame] * padding + frames
            actions = [self._one_hot[4]] * padding + actions

        frames = torch.stack(frames)  # [seq_len, 3, H, W]
        actions = torch.stack(actions)  # [seq_len, 5]

        # Flatten frames: obs = first seq_len-1 frames, target = last frame
        B, C, H, W = frames.shape
        frames_flat = frames.view(-1, H, W)  # [seq_len*3, H, W]
        obs = frames_flat[:-C, :, :]  # [(seq_len-1)*3, H, W]
        img = frames_flat[-C:, :, :]  # [3, H, W]

        # Reshape actions: [1, seq_len-1, 5]
        y = actions[:-1].unsqueeze(0)  # obs actions only
        y_mask = torch.ones(1, y.shape[1])

        result = {
            'obs': obs,
            'img': img,
            'y': y,
            'y_mask': y_mask,
            'data_info': {'episode': self.ds[idx]['episode'], 'done': self.ds[idx]['done']}
        }

        # Attach precomputed latent for target frame
        if self.load_vae_feat and self.latents is not None:
            result['img_latent'] = self.latents[idx]

        return result
