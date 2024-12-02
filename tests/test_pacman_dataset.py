import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusion.data.datasets.pacman_data import PacmanDataset, PacmanDatasetMS
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST
import numpy as np

class TestPacmanDataset(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.dataset = PacmanDataset(
            transform=self.transform,
            sequence_length=16,
            buffer_size=100
        )
    
    def test_dataset_batch_structure(self):
        """Test that dataset returns batches with the correct structure for training."""
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
        # After DataLoader batching: [B, C, H, W]
        self.assertEqual(batch['img'].shape[0], batch_size)  # Batch size
        self.assertEqual(len(batch['img'].shape), 4)  # [B, C, H, W]
        self.assertEqual(batch['img'].dtype, torch.float32)
        
        # After DataLoader batching: [B, seq_len, 5]
        self.assertEqual(batch['y'].shape[0], batch_size)  # Batch size
        self.assertEqual(len(batch['y'].shape), 3)  # [B, seq_len, 5]
        self.assertEqual(batch['y'].shape[-1], 5)  # One-hot encoded actions
        
        # After DataLoader batching: [B, 1, seq_len]
        self.assertEqual(batch['y_mask'].shape[0], batch_size)
        
        # Check data_info structure
        self.assertIn('frames', batch['data_info'])
        self.assertIn('episode', batch['data_info'])
        self.assertIn('done', batch['data_info'])
        
        # After DataLoader batching: [B, seq_len, C, H, W]
        frames_shape = batch['data_info']['frames'].shape
        self.assertEqual(len(frames_shape), 5)
        self.assertEqual(frames_shape[0], batch_size)
        self.assertEqual(frames_shape[1], 16)  # sequence_length
        self.assertEqual(frames_shape[2], 3)   # channels
    
    def test_dataset_batch_iteration(self):
        """Test that we can iterate through the dataset with different batch sizes."""
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True
            )
            
            # Get first batch
            batch = next(iter(dataloader))
            
            # Check batch sizes
            self.assertEqual(batch['img'].shape[0], batch_size)
            self.assertEqual(batch['y'].shape[0], batch_size)
            self.assertEqual(batch['y_mask'].shape[0], batch_size)
            self.assertEqual(batch['data_info']['frames'].shape[0], batch_size)
    
    def test_action_encoding(self):
        """Test that actions are properly one-hot encoded."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check action tensor shape and type
        self.assertEqual(batch['y'].dtype, torch.float32)
        self.assertEqual(len(batch['y'].shape), 3)  # [B, seq_len, 5]
        self.assertEqual(batch['y'].shape[-1], 5)  # 5 action classes
        
        # Verify one-hot encoding properties
        actions = batch['y'][0]  # [seq_len, 5]
        for action_vector in actions:
            # Each vector should sum to 1 (one-hot)
            self.assertEqual(action_vector.sum().item(), 1.0)
            # Each element should be either 0 or 1
            self.assertTrue(torch.all((action_vector == 0) | (action_vector == 1)))
    
    def test_sequence_padding(self):
        """Test that sequences are properly padded when needed."""
        # Create dataset with longer sequence length to test padding
        long_seq_dataset = PacmanDataset(
            transform=self.transform,
            sequence_length=32,  # Longer sequence
            buffer_size=50
        )
        
        dataloader = DataLoader(
            long_seq_dataset,
            batch_size=1,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check sequence length after batching
        self.assertEqual(batch['data_info']['frames'].shape[1], 32)
        self.assertEqual(batch['y'].shape[1], 32)
        
        # Check that actions for padded frames are one-hot encoded zeros
        first_action = batch['y'][0, 0]  # First action in sequence
        self.assertEqual(first_action[4], 1)  # First class should be 1 (padding)
        self.assertTrue(torch.all(first_action[:4] == 0))  # Rest should be 0

if __name__ == '__main__':
    unittest.main()
