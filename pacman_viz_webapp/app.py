import os
import json
import uuid
import numpy as np
import torch
from flask import Flask, render_template, jsonify, request, send_file
from PIL import Image
import io
import base64
from pacman_data_copy import PacmanDataset
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variables
dataset = None
sequence_buffer = []
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)

# Action labels
# Swapping LEFT and RIGHT indices (0 and 1)
ACTION_LABELS = ["LEFT", "RIGHT", "UP", "DOWN", "NO_ACTION"]

def init_dataset(sequence_length=8, resolution=128, stagger_offset=1):
    """Initialize the dataset with the given parameters"""
    global dataset, stagger_offset_global
    
    # Store the stagger offset as a global variable
    stagger_offset_global = stagger_offset
    
    # Calculate a safe context buffer to handle any offset
    # We'll fetch longer sequences than what we display to ensure we have enough context
    # for both past and future actions (negative and positive offsets)
    context_buffer = 10  # Extra frames for context (5 before, 5 after)
    actual_sequence_length = sequence_length + context_buffer
    
    if dataset is None or dataset.sequence_length != actual_sequence_length:
        dataset = PacmanDataset(
            transform=None, 
            load_vae_feat=False, 
            load_text_feat=False,
            sequence_length=actual_sequence_length, 
            resolution=resolution, 
            buffer_size=20,  # Larger buffer for smoother experience
            prefetch_factor=2
        )
    return dataset

def normalize_image_for_visualization(image_tensor_hwc):
    """Normalizes a single image tensor (H, W, C) to [0,1] float32 for display"""
    frame_np = image_tensor_hwc.numpy()
    min_val = np.min(frame_np)
    max_val = np.max(frame_np)
    delta = max_val - min_val
    if delta < 1e-7:  # Handle constant or near-constant frames
        normalized_frame = np.full_like(frame_np, 0.5, dtype=np.float32)  # Grey if constant
    else:
        normalized_frame = (frame_np - min_val) / delta
    return np.clip(normalized_frame, 0, 1).astype(np.float32)

def tensor_to_base64(tensor):
    """Convert a tensor to a base64 encoded image"""
    if tensor.dim() == 3:  # [H, W, C]
        img_array = (normalize_image_for_visualization(tensor) * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    return None

def get_cached_sequence(sequence_id):
    """Try to get a sequence from cache"""
    cache_file = os.path.join(cache_dir, f"sequence_{sequence_id}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def cache_sequence(sequence_data):
    """Cache a processed sequence for future use"""
    # Generate a unique ID for this sequence
    sequence_id = str(uuid.uuid4())
    
    # Save serializable data (exclude tensors)
    cache_data = {
        'id': sequence_id,
        'frames_b64': [tensor_to_base64(frame) for frame in sequence_data['frames']],
        'target_b64': tensor_to_base64(sequence_data['target']),
        'action_indices': sequence_data['action_indices']  # Store raw indices instead of formatted strings
    }
    
    # Save to cache
    cache_file = os.path.join(cache_dir, f"sequence_{sequence_id}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return cache_data

def process_sequence(processed_batch, batch_idx=0, display_frames=8):
    """Process a single sequence from the batch"""
    
    # Extract components
    obs = processed_batch['obs']         # Observation frames [(seq_len-1)*C, H, W]
    img = processed_batch['img']         # Target image [C, H, W]
    y = processed_batch['y']             # Actions [1, seq_len-1, 5]
    
    # Get sequence length from data shape
    C = 3  # RGB channels
    num_obs_frames = obs.shape[0] // C
    
    # Calculate which frames to display
    # We want to ensure the target frame is the one right after the displayed frames
    # So we'll end our display range right before the target frame
    display_frames = min(display_frames, num_obs_frames - 1)  # Ensure we leave at least 1 frame for target
    end_idx = num_obs_frames - 1  # Last frame before target
    start_idx = max(0, end_idx - display_frames + 1)  # Calculate start to get desired number of frames
    
    # Reshape observation frames - get all frames for context
    all_frames = []
    for i in range(num_obs_frames):
        frame = obs[i*C:(i+1)*C].permute(1, 2, 0)  # [H, W, C]
        all_frames.append(frame)
    
    # Select the frames to display (ending right before target)
    display_frames = all_frames[start_idx:end_idx+1]  # +1 because end is exclusive
    
    # Process target frame
    target = img.permute(1, 2, 0)  # [H, W, C]
    
    # Get action indices for all frames (for context)
    all_action_indices = y[0, :, :].argmax(dim=1).tolist()
    
    return {
        'frames': display_frames,                # Only the frames to display
        'target': target,
        'action_indices': all_action_indices,    # All action indices for context
        'display_range': {                       # Information about the display range
            'start': start_idx,
            'end': end_idx,
            'total': num_obs_frames,
            'target_idx': num_obs_frames         # Index of the target frame
        }
    }

def get_next_sequence():
    """Get the next sequence from the dataset"""
    global dataset
    
    if dataset is None:
        init_dataset()
    
    # Use iterator to get next item
    iterator = iter(dataset)
    try:
        batch = next(iterator)
        return process_sequence(batch)
    except StopIteration:
        return None
    except Exception as e:
        print(f"Error getting next sequence: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main visualization page"""
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def initialize_dataset():
    """Initialize the dataset with parameters from the request"""
    params = request.get_json()
    sequence_length = int(params.get('sequence_length', 8))
    resolution = int(params.get('resolution', 128))
    stagger_offset = int(params.get('stagger_offset', 1))
    
    # Ensure sequence_length is at least 8 frames
    sequence_length = max(8, sequence_length)
    
    init_dataset(sequence_length, resolution, stagger_offset)
    return jsonify({
        'status': 'success', 
        'message': 'Dataset initialized',
        'stagger_offset': stagger_offset,
        'sequence_length': sequence_length
    })

@app.route('/api/next_sequence', methods=['GET'])
def fetch_next_sequence():
    """Fetch and return the next sequence"""
    sequence_data = get_next_sequence()
    if sequence_data:
        # Cache the sequence and return serializable data
        cache_data = cache_sequence(sequence_data)
        return jsonify({'status': 'success', 'data': cache_data})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to get next sequence'})

@app.route('/api/cached_sequences', methods=['GET'])
def list_cached_sequences():
    """List all cached sequences"""
    cached_files = [f for f in os.listdir(cache_dir) if f.startswith('sequence_') and f.endswith('.pkl')]
    sequences = []
    
    for file in cached_files:
        sequence_id = file.replace('sequence_', '').replace('.pkl', '')
        cache_data = get_cached_sequence(sequence_id)
        if cache_data:
            sequences.append({
                'id': sequence_id,
                'preview': cache_data['frames_b64'][0]  # First frame as preview
            })
    
    return jsonify({'status': 'success', 'sequences': sequences})

@app.route('/api/sequence/<sequence_id>', methods=['GET'])
def get_sequence(sequence_id):
    """Get a specific cached sequence"""
    cache_data = get_cached_sequence(sequence_id)
    if cache_data:
        return jsonify({'status': 'success', 'data': cache_data})
    else:
        return jsonify({'status': 'error', 'message': 'Sequence not found'}, 404)

if __name__ == '__main__':
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    app.run(debug=True, port=5000)
