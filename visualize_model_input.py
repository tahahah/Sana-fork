import torch
import matplotlib.pyplot as plt
import numpy as np
from pacman_data_copy import PacmanDataset
from torch.utils.data import DataLoader
from matplotlib.widgets import Button

# --- Globals for managing DataLoader state and figures ---
dataloader_iterator = None
dataloader_instance = None
# To keep track of figures for each batch, so they can be closed
# This list will store figure objects.
active_figures = []

def normalize_image_for_visualization(image_tensor_hwc):
    """Normalizes a single image tensor (H, W, C) to [0,1] float32 for imshow."""
    frame_np = image_tensor_hwc.numpy()
    min_val = np.min(frame_np)
    max_val = np.max(frame_np)
    delta = max_val - min_val
    if delta < 1e-7:  # Handle constant or near-constant frames
        # Normalize to 0.5 (grey) if constant, ensuring it's float32
        normalized_frame = np.full_like(frame_np, 0.5, dtype=np.float32)
    else:
        normalized_frame = (frame_np - min_val) / delta
    return np.clip(normalized_frame, 0, 1).astype(np.float32)

def plot_single_processed_sequence(processed_sequence_data, sequence_idx_in_dl_batch, total_sequences_in_dl_batch, global_sequence_length):
    """
    Plots a single processed sequence (obs, target img, actions) in a new figure.
    'processed_sequence_data' is a dict for one item from the DataLoader's batch.
    'global_sequence_length' is the PacmanDataset's sequence_length parameter.
    """
    # User's preferred action labels (from Step 118)
    action_labels = ["RIGHT", "LEFT", "UP", "DOWN", "NO_ACTION"]

    # Extract data for this single sequence
    obs_tensor_chw_total = processed_sequence_data['obs']  # Shape: [(L-1)*C, H, W]
    target_img_tensor_chw = processed_sequence_data['img']  # Shape: [C, H, W]
    actions_tensor_1ls = processed_sequence_data['y']      # Shape: [1, L-1, NumActions]

    C = 3  # Assuming RGB
    num_obs_frames = global_sequence_length - 1
    
    # obs_tensor_chw_total has shape [(num_obs_frames)*C, H, W]
    # Reshape to [num_obs_frames, C, H, W]
    H_dim = obs_tensor_chw_total.shape[1]
    W_dim = obs_tensor_chw_total.shape[2]
    obs_reshaped_lchw = obs_tensor_chw_total.view(num_obs_frames, C, H_dim, W_dim)
    
    # Permute to [num_obs_frames, H, W, C] for visualization
    obs_frames_torch_lhwc = obs_reshaped_lchw.permute(0, 2, 3, 1).cpu()

    obs_frames_viz = [normalize_image_for_visualization(frame) for frame in obs_frames_torch_lhwc]
    target_img_viz = normalize_image_for_visualization(target_img_tensor_chw.permute(1, 2, 0).cpu())

    # Actions: from [1, L-1, NumActions] to [L-1] indices
    action_indices_np = torch.argmax(actions_tensor_1ls.squeeze(0), dim=1).cpu().numpy()

    # Create a new figure for this sequence
    fig, axes = plt.subplots(3, num_obs_frames, figsize=(num_obs_frames * 2.5, 7.5))
    if num_obs_frames == 1: # Adjust axes array shape if only one obs frame
        axes = axes.reshape(3,1)

    for i in range(num_obs_frames):
        # Row 1: Observation frames
        axes[0, i].imshow(obs_frames_viz[i])
        axes[0, i].set_title(f"Obs Frame {i}")
        axes[0, i].axis('off')
        
        # Row 2: Actions (shown below corresponding obs frame)
        axes[1, i].imshow(obs_frames_viz[i]) # Show frame again for visual context
        action_int_idx = int(action_indices_np[i])
        action_name = action_labels[action_int_idx] if action_int_idx < len(action_labels) else "Unknown"
        axes[1, i].set_title(f"Action: {action_name}")
        axes[1, i].axis('off')
        
        # Row 3: Target Frame (repeated for alignment)
        axes[2, i].imshow(target_img_viz)
        axes[2, i].set_title(f"Target Img")
        axes[2, i].axis('off')
        
    fig.suptitle(f"DataLoader Batch - Sequence {sequence_idx_in_dl_batch + 1}/{total_sequences_in_dl_batch}\n(Dataset seq_len={global_sequence_length})", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect for suptitle
    return fig # Return the figure object

def on_next_batch_click(event):
    """Callback for the 'Next Batch' button."""
    global dataloader_iterator, dataloader_instance, active_figures, dataset_sequence_length

    # Close all previously opened sequence figures
    for fig in active_figures:
        plt.close(fig)
    active_figures.clear()

    if dataloader_iterator is None:
        print("DataLoader iterator not initialized.")
        return

    print("Fetching next batch from DataLoader...")
    try:
        # This 'dl_batch' contains 'dataloader_batch_size' number of processed sequences
        dl_batch = next(dataloader_iterator)
    except StopIteration:
        print("DataLoader exhausted. Re-initializing iterator.")
        dataloader_iterator = iter(dataloader_instance) # Re-initialize
        try:
            dl_batch = next(dataloader_iterator)
        except StopIteration:
            print("DataLoader is empty even after re-initialization. Cannot fetch batch.")
            return
    
    # The actual number of sequences in this DataLoader batch
    # This should match dataloader_instance.batch_size
    num_sequences_in_dl_batch = dl_batch['img'].shape[0] 
    
    print(f"Fetched DataLoader batch containing {num_sequences_in_dl_batch} sequences.")
    print(f"  Shape of 'img' tensor in DL batch: {dl_batch['img'].shape}") # e.g., [seq_len, C, H, W]
    print(f"  Shape of 'obs' tensor in DL batch: {dl_batch['obs'].shape}") # e.g., [seq_len, (L-1)*C, H, W]
    print(f"  Shape of 'y' tensor in DL batch: {dl_batch['y'].shape}")       # e.g., [seq_len, 1, L-1, 5]

    # Iterate through each processed sequence within this DataLoader batch
    for i in range(num_sequences_in_dl_batch):
        single_processed_sequence_data = {
            'img': dl_batch['img'][i],
            'obs': dl_batch['obs'][i],
            'y': dl_batch['y'][i],
            'y_mask': dl_batch['y_mask'][i]
            # Add 'data_info' if necessary: 'data_info': dl_batch['data_info'][i] (if it's a list of dicts)
        }
        
        new_fig = plot_single_processed_sequence(
            single_processed_sequence_data,
            sequence_idx_in_dl_batch=i,
            total_sequences_in_dl_batch=num_sequences_in_dl_batch,
            global_sequence_length=dataset_sequence_length # Pass the dataset's sequence_length
        )
        active_figures.append(new_fig)
    
    plt.show() # This will draw all newly created figures

# Store dataset's sequence_length globally for access in callbacks
dataset_sequence_length = 0

def visualize_model_input(sequence_length_param=8, resolution_param=128):
    """Main function to set up dataset, dataloader, button, and initial plot."""
    global dataloader_iterator, dataloader_instance, active_figures, dataset_sequence_length

    dataset_sequence_length = sequence_length_param # Store for later use

    # User's PacmanDataset instantiation (from Step 118)
    ds = PacmanDataset(transform=None, load_vae_feat=False, load_text_feat=False,
                       sequence_length=sequence_length_param, resolution=resolution_param, 
                       buffer_size=10, prefetch_factor=1) # User's buffer/prefetch
    
    # User's DataLoader batch_size setting (from Step 118)
    # This means each item from dataloader_instance IS a batch of 'sequence_length_param' sequences
    dataloader_batch_size = sequence_length_param 
    dataloader_instance = DataLoader(ds, batch_size=dataloader_batch_size, num_workers=0, shuffle=False)
    dataloader_iterator = iter(dataloader_instance)
    
    # --- Create a separate, persistent figure for the button ---
    fig_button_control = plt.figure("Controls", figsize=(3, 1.5))
    plt.subplots_adjust(bottom=0.3)
    button_ax = plt.axes([0.15, 0.2, 0.7, 0.6]) # x, y, width, height
    bnext = Button(button_ax, 'Next DataLoader Batch')
    bnext.on_clicked(on_next_batch_click)
    # We don't add fig_button_control to active_figures so it's not closed by on_next_batch_click

    # Initial fetch and display of the first DataLoader batch
    print("Displaying initial batch...")
    on_next_batch_click(None) # Trigger initial plot
    
    plt.show() # Keeps all figures (including button) alive until manually closed

if __name__ == '__main__':
    # The 'batch_size' parameter for visualize_model_input is not directly used for DataLoader's batch_size here,
    # as per user's existing DataLoader setup which uses sequence_length for its batch_size.
    visualize_model_input(sequence_length_param=8, resolution_param=128)