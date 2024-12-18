import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusion.data.datasets.pacman_data import PacmanDataset, PacmanDatasetMS, load_dataset
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
import tempfile
import shutil
import logging

class MockVAE:
    def __init__(self):
        self.encode_count = 0
        self.parameters = lambda: iter([torch.nn.Parameter(torch.randn(1))])
        
    def encode(self, x):
        """
        Input: [B, C, H, W] where H=W=512
        Output: Object with latent_dist that samples tensor of shape [B, 32, 16, 16]
        32x compression: spatial compression from 512x512 to 16x16, 
        with 32 channels instead of 3
        """
        self.encode_count += 1
        B, C, H, W = x.shape
        # Create a mock latent tensor with correct dimensions
        latent = torch.randn(B, 32, H//32, W//32)
        
        class MockLatentDist:
            def __init__(self, latent):
                self.latent = latent
                
            def sample(self):
                return self.latent
        
        class MockOutput:
            def __init__(self, latent_dist):
                self.latent_dist = latent_dist
                
            def cpu(self):
                return self.latent_dist.sample()  # Add cpu method to return tensor
        
        return MockOutput(MockLatentDist(latent))
        
    def to(self, device):
        return self

@patch('diffusion.data.datasets.pacman_data.load_dataset')
class TestPacmanDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config = MagicMock()
        self.config.work_dir = self.test_dir
        self.config.model = MagicMock()
        self.config.model.mixed_precision = "no"
        
        # Setup transform and mock VAE
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),  # Ensure consistent size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mock_vae = MockVAE()

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
        
    def _create_dataset(self, mock_load_dataset, with_vae=True, raw_mode=False):
        # Use the actual dataset
        mock_load_dataset.return_value = load_dataset(
            "Tahahah/PacmanDataset_3",
            split="train",
            verification_mode="no_checks",
            streaming=True
        )
        
        return PacmanDataset(
            transform=self.transform,
            sequence_length=16,
            buffer_size=100,
            vae=self.mock_vae if with_vae else None,
            config=self.config,
            raw_mode=raw_mode  # New parameter for raw image output
        )

    def test_dataset_batch_structure(self, mock_load_dataset):
        """Test that dataset returns batches with the correct structure for training."""
        self.dataset = self._create_dataset(mock_load_dataset)
        batch_size = 4
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check basic structure required by training script
        self.assertIn('img', batch)
        self.assertIn('y', batch)
        self.assertIn('y_mask', batch)
        self.assertIn('data_info', batch)
        
        # Check tensor shapes and types
        self.assertEqual(batch['img'].shape[0], batch_size)  # Batch size
        logging.warn(f"Batch img shape: {batch['img'].shape}")
        self.assertEqual(len(batch['img'].shape), 4)  # [B, 32, 16, 16] for VAE encoded
        self.assertEqual(batch['img'].dtype, torch.float32)
        
        # Check action shape: [B, seq_len, 5] as per requirements
        self.assertEqual(batch['y'].shape, (batch_size, 1, self.dataset.sequence_length-1, 5))
        self.assertTrue(torch.all(batch['y'].sum(dim=-1) == 1))  # One-hot check
        
        # Check mask shape: [B, 1, seq_len]
        self.assertEqual(batch['y_mask'].shape[0], batch_size)
        
        # Check data_info structure
        self.assertIn('episode', batch['data_info'])
        self.assertIn('done', batch['data_info'])
    
    def test_vae_encoding_shape(self, mock_load_dataset):
        """Test that VAE encoding produces correct output shape with 32x compression."""
        self.dataset = self._create_dataset(mock_load_dataset)
        dataloader = DataLoader(self.dataset, batch_size=2)
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('img', batch)
        self.assertIn('y', batch)
        self.assertIn('y_mask', batch)
        
        # Check shapes according to Pacman_Pipeline.md requirements
        # VAE output shape: [batch_size, 32, 16, 16]
        batch_size = 2
        expected_shape = (batch_size, 32*(self.dataset.sequence_length - 1), 16, 16)
        self.assertEqual(batch['obs'].shape, expected_shape)
        
    def test_frame_caching(self, mock_load_dataset):
        """Test that frames are properly cached and reused."""
        self.dataset = self._create_dataset(mock_load_dataset)
        # Reset encode count
        self.mock_vae.encode_count = 0
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0
        )
        
        # Get first batch - should encode all frames
        batch1 = next(iter(dataloader))
        initial_encode_count = self.mock_vae.encode_count
        
        # Get second batch - should reuse some cached frames
        batch2 = next(iter(dataloader))
        second_encode_count = self.mock_vae.encode_count - initial_encode_count
        
        # Should encode fewer frames in second batch due to caching
        self.assertLess(second_encode_count, initial_encode_count)
        
    def test_cache_size_limit(self, mock_load_dataset):
        """Test that cache size is properly limited."""
        # Create dataset with small cache size for faster testing
        self.dataset = PacmanDataset(
            transform=self.transform,
            sequence_length=4,  # Smaller sequence length
            buffer_size=10,     # Smaller buffer
            vae=self.mock_vae,
            config=self.config
        )
        self.dataset._cache_size = 5  # Set small cache size
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0
        )
        
        # Process enough batches to fill cache
        for _ in range(3):  # Just need a few iterations
            try:
                next(iter(dataloader))
            except StopIteration:
                break
            
        # Check cache size hasn't exceeded limit
        self.assertLessEqual(
            len(self.dataset._encoded_frames_cache),
            self.dataset._cache_size
        )
        
    def test_blank_frame_caching(self, mock_load_dataset):
        """Test that blank frames are properly cached and reused."""
        self.dataset = self._create_dataset(mock_load_dataset)
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0
        )
        
        # Reset encode count
        self.mock_vae.encode_count = 0
        
        # Get multiple batches that require padding
        batch1 = next(iter(dataloader))
        initial_blank_latent = self.dataset.blank_latent
        
        # Get another batch
        batch2 = next(iter(dataloader))
        second_blank_latent = self.dataset.blank_latent
        
        # Should reuse the same blank latent
        self.assertTrue(torch.equal(initial_blank_latent, second_blank_latent))
        
    def test_no_vae_shape(self, mock_load_dataset):
        """Test dataset output shape when VAE is disabled."""
        self.dataset = self._create_dataset(mock_load_dataset, with_vae=False)
        dataloader = DataLoader(self.dataset, batch_size=2)
        batch = next(iter(dataloader))
        
        # Check batch structure
        self.assertIn('img', batch)
        self.assertIn('y', batch)
        self.assertIn('y_mask', batch)
        
        # Check shapes for non-VAE case
        # Dataset output: [seq_len, C, H, W]
        # DataLoader adds batch dim: [B, C, H, W]
        batch_size = 2
        self.assertEqual(len(batch['img'].shape), 4)
        self.assertEqual(batch['img'].shape[0], batch_size)
        self.assertEqual(batch['img'].shape[1], 3)  # Adjusted for channels
        self.assertEqual(batch['img'].shape[2:], (512, 512))

        # DataLoader adds batch dim: [B, C*(seq_len-1), H, W]
        self.assertEqual(len(batch['obs'].shape), 4)
        self.assertEqual(batch['obs'].shape[0], batch_size)
        self.assertEqual(batch['obs'].shape[1], 3 * (self.dataset.sequence_length - 1))  # Adjusted for channels
        self.assertEqual(batch['obs'].shape[2:], (512, 512))

    def test_raw_mode_batch_structure(self, mock_load_dataset):
        """
        Test that dataset returns batches with raw images in the correct structure.
        This fulfills the requirement from Gameplan.md for raw image output:
        [b, 3*seq_length, h, w] for raw image frames
        """
        self.dataset = self._create_dataset(mock_load_dataset, with_vae=False, raw_mode=True)
        batch_size = 4
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check basic structure
        self.assertIn('img', batch)
        self.assertIn('y', batch)
        self.assertIn('y_mask', batch)
        self.assertIn('data_info', batch)
        
        # Check raw image tensor shapes
        # Should be [B, 3, H, W] for raw images
        self.assertEqual(batch['img'].shape[0], batch_size)  # Batch size
        self.assertEqual(batch['img'].shape[1], 3)  # Channels
        self.assertEqual(batch['img'].shape[2], 512)  # Height
        self.assertEqual(batch['img'].shape[3], 512)  # Width
        self.assertEqual(batch['img'].dtype, torch.float32)

        # Check obs tensor shapes
        # Should be [B, 3*(seq_length-1), H, W]
        self.assertEqual(batch['obs'].shape[0], batch_size)  # Batch size
        self.assertEqual(batch['obs'].shape[1], 3 * (self.dataset.sequence_length - 1))  # Channels * seq_length-1
        self.assertEqual(batch['obs'].shape[2], 512)  # Height
        self.assertEqual(batch['obs'].shape[3], 512)  # Width
        self.assertEqual(batch['obs'].dtype, torch.float32)
        
        # Verify pixel value range after normalization
        self.assertTrue(torch.all(batch['img'] >= -1.0))
        self.assertTrue(torch.all(batch['img'] <= 1.0))

    def test_backward_compatibility(self, mock_load_dataset):
        """
        Test that the dataset maintains backward compatibility when raw_mode is False.
        This ensures existing training pipelines continue to work.
        """
        # Test with raw_mode=False (default behavior)
        self.dataset = self._create_dataset(mock_load_dataset, with_vae=True, raw_mode=False)
        batch_size = 4
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Should still output VAE encoded format
        expected_shape = (batch_size, 32*(self.dataset.sequence_length-1), 16, 16)
        self.assertEqual(batch['obs'].shape, expected_shape)

        
        expected_shape = (batch_size, 32, 16, 16)
        self.assertEqual(batch['img'].shape, expected_shape)

    def test_sequence_processing(self, mock_load_dataset):
        """
        Test that sequences are properly processed in raw mode.
        Verifies that frames are correctly ordered and concatenated.
        """
        self.dataset = self._create_dataset(mock_load_dataset, with_vae=False, raw_mode=True)
        batch_size = 1
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check sequence structure
        img_sequence = batch['obs'][0]  # Take first batch
        num_frames = self.dataset.sequence_length-1
        
        # Should be able to split into individual frames
        frames = img_sequence.reshape(num_frames, 3, 512, 512)
        
        # Each frame should be a valid image
        for frame in frames:
            self.assertEqual(frame.shape, (3, 512, 512))
            self.assertTrue(torch.all(frame >= -1.0))
            self.assertTrue(torch.all(frame <= 1.0))

    
    
    def test_obs_and_x(self, mock_load_dataset):
        self.dataset = self._create_dataset(mock_load_dataset, with_vae=False)
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            num_workers=0
        )

        batch = next(iter(dataloader))
        self.assertEqual(batch['obs'].shape, (4, 3*(self.dataset.sequence_length-1), 512, 512))
        self.assertEqual(batch['img'].shape, (4, 3, 512, 512))
        self.assertEqual(batch['img'].shape[1] + batch['obs'].shape[1], 3*self.dataset.sequence_length)


if __name__ == '__main__':
    unittest.main()
