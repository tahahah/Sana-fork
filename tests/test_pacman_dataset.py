import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusion.data.datasets.pacman_data import PacmanDataset, PacmanDatasetMS

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
        
    def test_dataset_batch_iteration(self):
        """Test that we can iterate through the dataset with different batch sizes."""
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True
            )
            
            # Get first batch
            batch = next(iter(dataloader))
            
            # Check batch structure
            self.assertIn('img', batch)
            self.assertIn('y', batch)
            self.assertIn('y_mask', batch)
            self.assertIn('data_info', batch)
            
            # Check batch sizes
            self.assertEqual(batch['img'].shape[0], batch_size)
            self.assertEqual(batch['y'].shape[0], batch_size)
            self.assertEqual(batch['y_mask'].shape[0], batch_size)
            
            # Check sequence length
            self.assertEqual(batch['y'].shape[2], 16)  # sequence_length
            
            # Check image dimensions
            self.assertEqual(batch['img'].shape[1:], (3, 256, 256))
            
    def test_multi_scale_dataset(self):
        """Test the multi-scale version of the dataset."""
        ms_dataset = PacmanDatasetMS(
            transform=self.transform,
            sequence_length=16,
            buffer_size=100,
            aspect_ratio_type="ASPECT_RATIO_512"
        )
        
        dataloader = DataLoader(
            ms_dataset,
            batch_size=4,
            num_workers=2,
            pin_memory=True
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        # Check that it has the same structure as base dataset
        self.assertIn('img', batch)
        self.assertIn('y', batch)
        self.assertIn('y_mask', batch)
        self.assertIn('data_info', batch)

if __name__ == '__main__':
    unittest.main()
