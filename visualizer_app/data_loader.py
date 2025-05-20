# This file will contain functions to load and process data.

import torch
import torchvision.transforms as T
from PIL import Image
import os
import shutil
from collections import deque
from datasets import load_dataset
import logging

# Constants
UI_ACTION_LABELS = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "NO_ACTION"}
TEMP_FRAME_DIR = "visualizer_app/static/temp_frames"

# Helper functions
def convert_to_rgb(img):
    return img.convert("RGB")

def make_square(image, min_size=256, fill_color=(0, 0, 0, 0)):
    """Pads a PIL image to make it square.

    Args:
        image: PIL.Image.Image, the input image.
        min_size: int, the minimum side length of the square.
        fill_color: tuple, the color to use for padding.
    Returns:
        PIL.Image.Image, the padded square image.
    """
    x, y = image.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def rotate_90_clockwise(img):
    return img.rotate(90, expand=True)

class VisualizationDataLoader:
    def __init__(self, resolution=128, hf_dataset_name="Tahahah/PacmanDataset_3"):
        self.resolution = resolution
        self.raw_dataset = load_dataset(hf_dataset_name, split="train", streaming=True)
        
        self.transform = T.Compose([
            T.Lambda(convert_to_rgb),
            T.Lambda(make_square),
            T.Resize([resolution, resolution]),
            T.RandomHorizontalFlip(p=1.0), # Equivalent to transforms.functional.hflip
            T.Lambda(rotate_90_clockwise),
            T.ToTensor() # Scales to [0,1]
        ])
        
        self.dataset_iterator = iter(self.raw_dataset)
        self.action_list_size = 5 # Number of possible actions
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Ensure TEMP_FRAME_DIR exists and is empty
        if os.path.exists(TEMP_FRAME_DIR):
            shutil.rmtree(TEMP_FRAME_DIR)
        os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
        self.logger.info(f"Temporary frame directory created at {TEMP_FRAME_DIR}")

    def _fetch_raw_sequence(self, num_items):
        raw_sequence = []
        for _ in range(num_items):
            try:
                item = next(self.dataset_iterator)
                # Ensure 'image' and 'action' keys exist
                if 'image' in item and 'action' in item:
                    raw_sequence.append({'frame_pil': item['image'], 'action': item['action']})
                else:
                    self.logger.warning("Skipping item due to missing 'image' or 'action' key.")
                    # Optionally, try fetching another item to compensate
                    # This might lead to an infinite loop if the dataset is consistently bad
            except StopIteration:
                self.logger.info("Dataset iterator exhausted. Resetting.")
                self.dataset_iterator = iter(self.raw_dataset)
                # Try fetching again after reset
                try:
                    item = next(self.dataset_iterator)
                    if 'image' in item and 'action' in item:
                        raw_sequence.append({'frame_pil': item['image'], 'action': item['action']})
                    else:
                        self.logger.warning("Skipping item after reset due to missing keys.")
                except StopIteration: # Should not happen with a non-empty dataset
                    self.logger.error("Dataset empty even after reset. Returning partial sequence.")
                    break 
            except Exception as e:
                self.logger.error(f"Error fetching item from dataset: {e}")
                # Decide how to handle other errors, e.g., skip item, re-raise, etc.
                # For now, we'll skip the item
        return raw_sequence

    def get_visualization_data(self, num_display_frames, context_window_half):
        total_items_to_fetch = num_display_frames + 2 * context_window_half
        raw_data = self._fetch_raw_sequence(total_items_to_fetch)

        if len(raw_data) < total_items_to_fetch:
            self.logger.warning(f"Could only fetch {len(raw_data)} items, requested {total_items_to_fetch}. Visualization might be incomplete.")
            # Adjust num_display_frames if not enough data for full context for all display frames
            # This logic ensures we don't try to access raw_data out of bounds
            effective_displayable_frames = len(raw_data) - 2 * context_window_half
            if effective_displayable_frames < num_display_frames:
                self.logger.warning(f"Adjusting num_display_frames from {num_display_frames} to {max(0, effective_displayable_frames)}")
                num_display_frames = max(0, effective_displayable_frames)


        processed_frames_data = []
        
        # Ensure TEMP_FRAME_DIR is clean before saving new frames
        if os.path.exists(TEMP_FRAME_DIR):
            shutil.rmtree(TEMP_FRAME_DIR)
        os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

        for i in range(num_display_frames):
            # Index for the current frame in raw_data that will be displayed
            current_frame_raw_idx = i + context_window_half
            
            # Check if the current_frame_raw_idx is valid
            if current_frame_raw_idx >= len(raw_data):
                self.logger.warning(f"Skipping frame {i} as data is not available. Index {current_frame_raw_idx} out of bounds for raw_data length {len(raw_data)}.")
                continue

            raw_item = raw_data[current_frame_raw_idx]
            frame_pil = raw_item['frame_pil']
            
            transformed_frame_tensor = self.transform(frame_pil)
            frame_pil_to_save = T.ToPILImage()(transformed_frame_tensor)
            
            frame_filename = f"frame_{i:03d}.png"
            frame_path = os.path.join(TEMP_FRAME_DIR, frame_filename)
            frame_pil_to_save.save(frame_path)
            
            image_url = f"/static/temp_frames/{frame_filename}" # FastAPI serves from static directory
            
            # Collect actions for the full context window relevant to this displayed frame
            # The context window slides with 'i'
            start_idx_for_actions = i 
            end_idx_for_actions = i + (2 * context_window_half) + 1 # +1 because slice is exclusive at end

            # Ensure action window does not go out of bounds of raw_data
            actual_end_idx_for_actions = min(end_idx_for_actions, len(raw_data))
            
            current_actions_raw = [item['action'] for item in raw_data[start_idx_for_actions:actual_end_idx_for_actions]]
            current_actions_labels = [UI_ACTION_LABELS.get(action, "UNKNOWN") for action in current_actions_raw]


            processed_frames_data.append({
                'frame_url': image_url,
                'actions': current_actions_labels, # These are actions for the whole window around the current displayed frame
                'original_frame_index_in_raw': current_frame_raw_idx 
            })
            
        return processed_frames_data
