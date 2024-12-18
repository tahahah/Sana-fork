# Pacman Diffusion Model - Inference Workflow

This document details the complete inference workflow of our Pacman Diffusion Model, including tensor shapes and the role of each component.

## Overview

Our model combines a HistoryEncoder with a SanaMS backbone to generate Pacman game frames conditioned on action sequences. The model uses classifier-free guidance (CFG) and DPM-Solver for efficient sampling.

## Detailed Workflow

### 1. Initial Input Setup

The model takes the following inputs during sampling:

```python
img: [B, S, C, H, W]      # Batch of image sequences
obs: [B, S, C, H, W]      # Batch of observations
actions: [B, S, A]        # Batch of actions (A=5 for Pacman)
action_masks: [B, S]      # Masks for actions
```

Where:
- B = batch size
- S = sequence length
- C = number of channels
- H, W = height and width
- A = number of actions (5 for Pacman)

### 2. Preprocessing

Actions are reshaped and null actions are created for classifier-free guidance:

```python
# Reshape actions for model input
actions = actions.unsqueeze(1)  # [B, 1, S, A]

# Create null actions for classifier-free guidance
null_action = torch.zeros(B, 1, S-1, 5)  # [B, 1, S-1, A]
null_action_mask = torch.ones(B, S-1)    # [B, S-1]
```

### 3. History Encoder

The HistoryEncoder processes the concatenated noisy input and observations:

```python
# Input tensors
x: [B, C, H, W]           # Noisy input image
obs: [B, C*S, H, W]       # Flattened sequence of observations

# Concatenate along channel dimension
combined = torch.cat([x, obs], dim=1)  # [B, C*(S+1), H, W]

# Output: Encoded history
encoded = history_encoder(combined)     # [B, 32, H, W]
```

Architecture:
- Series of Conv2d layers with BatchNorm and ReLU
- Channel dimension reduction: `C*(S+1)` → 512 → 256 → 128 → 32
- Spatial dimensions (H, W) preserved throughout

### 4. DPM-Solver Sampling

Setup for the DPM-Solver sampling process:

```python
# Input to DPM-Solver
z: [B, C, H, W]           # Initial noise
condition = actions       # [B, 1, S, A] - Actions as condition
uncondition = null_action # [B, 1, S, A] - Null actions for CFG

# Model kwargs
model_kwargs = {
    "data_info": {"img_hw": hw, "aspect_ratio": ar},
    "mask": None,
    "obs": obs  # [B, C*S, H, W] - Flattened observations
}
```

### 5. Forward Pass

The complete forward pass through the model:

```python
# PacmanDiffusionModel.forward
x = history_encoder(torch.cat([noisy_x, obs], dim=1))  # [B, 32, H, W]
output = sana(x, timestep, y, mask=mask, data_info=data_info)

# For DPM-Solver specifically (forward_with_dpmsolver)
model_out = forward(...)
return model_out.chunk(2, dim=1)[0]  # Return only predicted x_start, not variance
```

## Key Features

1. **Efficient History Encoding**: The HistoryEncoder compresses temporal history into a fixed-size latent representation.

2. **Classifier-Free Guidance**: Uses null actions as unconditional input for better control over generation.

3. **DPM-Solver Integration**: Efficient sampling process using the encoded history and action conditioning.

4. **Single-Pass Architecture**: The model predicts only x_start (not variance) when used with DPM-Solver.

## Benefits

This architecture provides:
- Effective encoding of temporal information from observation history
- Action-conditioned generation for controlled frame synthesis
- Enhanced control through classifier-free guidance
- Efficient sampling via DPM-Solver integration
