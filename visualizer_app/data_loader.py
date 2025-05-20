# This file will contain functions to load and process data.

import torch
import torchvision.transforms as T
from PIL import Image
import os
import shutil
# from collections import deque # Deque might not be needed anymore
# from datasets import load_dataset # Removed
from .pacman_dataset_copy import PacmanDataset # Added
import logging

# Constants
UI_ACTION_LABELS = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "NO_ACTION"}
TEMP_FRAME_DIR = "visualizer_app/static/temp_frames"

# Helper functions (convert_to_rgb, make_square, rotate_90_clockwise) are removed 
# as they are now encapsulated within pacman_dataset_copy.py.

class VisualizationDataLoader:
    def __init__(self, resolution=128, pacman_sequence_length=64): # Updated parameters
        self.resolution = resolution # Keep resolution as it's passed to PacmanDataset
        self.pacman_sequence_length = pacman_sequence_length # Store pacman_sequence_length

        # Instantiate the copied dataset
        # load_vae_feat and load_text_feat are False by default in PacmanDataset's new signature
        self.pacman_ds = PacmanDataset(
            resolution=self.resolution, 
            sequence_length=self.pacman_sequence_length,
            load_vae_feat=False # Explicitly False
        )
        
        self.dataset_iterator = iter(self.pacman_ds)
        self.action_list_size = 5 # Number of possible actions (remains relevant for UI_ACTION_LABELS)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO) # Ensures logger output is visible
        
        # Ensure TEMP_FRAME_DIR exists and is empty
        if os.path.exists(TEMP_FRAME_DIR):
            shutil.rmtree(TEMP_FRAME_DIR)
        os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
        self.logger.info(f"Temporary frame directory created at {TEMP_FRAME_DIR}")

    # _fetch_raw_sequence method removed

    def get_visualization_data(self): # Parameters updated
        # This method will now fetch and process *one* sequence 
        # from the PacmanDataset iterator.
        
        try:
            processed_sequence = next(self.dataset_iterator)
        except StopIteration:
            self.logger.info("PacmanDataset iterator exhausted. Resetting.")
            self.dataset_iterator = iter(self.pacman_ds) # Re-initialize iterator
            try:
                processed_sequence = next(self.dataset_iterator)
            except StopIteration:
                self.logger.error("Dataset empty even after reset. Returning empty data.")
                return [] # Should not happen with a streaming dataset that can be reset

        # Extract data from processed_sequence (referencing PacmanDataset structure)
        obs_tensor = processed_sequence['obs']  # Shape: [(L-1)*C, H, W]
        actions_tensor = processed_sequence['y'] # Shape: [1, L-1, NumActions]
        
        # Assuming C=3 (RGB). PacmanDataset's transform output is 3 channels.
        C = 3 
        # L is self.pacman_sequence_length. 'obs' has L-1 frames.
        num_frames_in_sequence = self.pacman_sequence_length - 1 
        
        # Dimensions H, W
        # obs_tensor shape is [ (L-1)*C, H, W ]. Example: [ (64-1)*3, 128, 128 ] = [189, 128, 128]
        H = obs_tensor.shape[1] 
        W = obs_tensor.shape[2]

        # Process Frames:
        # Reshape obs_tensor to (L-1, C, H, W)
        obs_frames_lchw = obs_tensor.view(num_frames_in_sequence, C, H, W)
        
        processed_frames_output = []
        
        # Ensure TEMP_FRAME_DIR is clean before saving new frames
        if os.path.exists(TEMP_FRAME_DIR):
            shutil.rmtree(TEMP_FRAME_DIR)
        os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

        # Process Actions:
        # actions_tensor shape: [1, L-1, NumActions_OneHot] e.g. [1, 63, 5]
        # We need to convert one-hot to class indices.
        action_indices = torch.argmax(actions_tensor.squeeze(0), dim=1).cpu().numpy() # Shape [L-1]
        action_labels_for_sequence = [UI_ACTION_LABELS.get(idx, "UNKNOWN") for idx in action_indices]

        # Loop for frames and actions
        for j in range(num_frames_in_sequence):
            frame_tensor_chw = obs_frames_lchw[j]
            
            # Convert to PIL: PacmanDataset's transform output is float16. ToPILImage expects float32 or uint8.
            frame_pil = T.ToPILImage()(frame_tensor_chw.cpu().float()) 
            
            frame_filename = f"frame_{j:03d}.png"
            frame_path = os.path.join(TEMP_FRAME_DIR, frame_filename)
            frame_pil.save(frame_path)
            
            image_url = f"/static/temp_frames/{frame_filename}"
            
            # Assemble output for this frame
            # 'actions' will be the full list of L-1 actions for this entire sequence.
            # 'original_frame_index_in_raw' will be the index of this frame within that action list.
            processed_frames_output.append({
                'frame_url': image_url,
                'actions': action_labels_for_sequence, 
                'original_frame_index_in_raw': j 
            })
            
        return processed_frames_output
