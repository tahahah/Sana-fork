import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import traceback
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

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

def visualize_sequence(sequence, sequence_idx, save_dir=".", stagger_offset=2):
    """
    Visualize a sequence from the Pacman dataset with staggered actions
    
    Args:
        sequence: A list of dictionaries containing frame_image and action
        sequence_idx: Index of the sequence for naming the output file
        save_dir: Directory to save the visualization
        stagger_offset: Number of frames to offset the actions by (backward)
    
    Returns:
        Path to the saved visualization image
    """
    # Define action labels
    action_labels = ["LEFT", "RIGHT", "UP", "DOWN", "NO_ACTION"]
    
    # Create a transform pipeline
    transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Lambda(make_square),  # Make the image square with padding
        transforms.Resize(512),          # Resize to 512x512
        transforms.functional.hflip,     # Horizontal mirror flip
        transforms.Lambda(rotate_90_clockwise),  # Rotate 90 degrees clockwise
        transforms.ToTensor(),
    ])
    
    # Get the number of frames in the sequence
    seq_len = len(sequence)
    
    # Create a figure to display the sequence
    fig, axes = plt.subplots(seq_len, 1, figsize=(10, seq_len * 3))
    if seq_len == 1:
        axes = [axes]
    
    plt.suptitle(f"Sequence {sequence_idx} - Pacman Frames (Actions from {stagger_offset} Frames Ago)", fontsize=16)
    
    # Plot each frame in the sequence
    for i, frame_data in enumerate(sequence):
        # Get the frame image
        frame_image = frame_data["frame_image"]
        
        # Get the staggered action (from a past frame)
        staggered_idx = i - stagger_offset
        if staggered_idx >= 0:
            action = sequence[staggered_idx]["action"]
            action_text = f"Action from {stagger_offset} frames ago: {action_labels[action]}"
        else:
            # For frames near the beginning where we don't have a past action
            action_text = "No past action available"
        
        # Transform the image
        frame_tensor = transform(frame_image)
        
        # Convert tensor to numpy for display
        frame_array = frame_tensor.permute(1, 2, 0).numpy()
        
        # Display the frame
        axes[i].imshow(frame_array)
        axes[i].set_title(f"Frame {i} - {action_text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"pacman_sequence_{sequence_idx}_staggered.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def main():
    try:
        # Create output directory if it doesn't exist
        output_dir = "pacman_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading Pacman dataset from Hugging Face...")
        
        # Load the dataset directly from Hugging Face
        dataset = load_dataset(
            "Tahahah/PacmanDataset_3", 
            split="train", 
            verification_mode="no_checks", 
            streaming=True
        )
        
        # Number of sequences to visualize
        num_sequences = 3
        # Number of frames per sequence - increase to account for staggering
        frames_per_sequence = 10  # Increased from 8 to allow for staggering
        # Stagger offset
        stagger_offset = 2
        
        saved_images = []
        
        print(f"Visualizing {num_sequences} sequences with {frames_per_sequence} frames each...")
        print(f"Actions will be staggered by {stagger_offset} frames")
        
        # Create an iterator
        dataset_iter = iter(dataset)
        
        # Get and visualize sequences
        for i in range(num_sequences):
            try:
                print(f"Processing sequence {i+1}/{num_sequences}")
                
                # Collect frames for this sequence
                sequence = []
                for j in range(frames_per_sequence):
                    try:
                        sample = next(dataset_iter)
                        sequence.append(sample)
                    except StopIteration:
                        print("Reached the end of the dataset")
                        break
                
                if not sequence:
                    break
                
                # Visualize the sequence with staggered actions
                saved_img = visualize_sequence(sequence, i+1, output_dir, stagger_offset)
                saved_images.append(saved_img)
                
                print(f"Saved visualization to {saved_img}")
                
            except Exception as e:
                print(f"Error processing sequence {i+1}: {str(e)}")
                print(traceback.format_exc())
        
        print("\nVisualization complete!")
        print(f"Saved images: {saved_images}")
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
