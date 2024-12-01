## Data Processing Pipeline

The goal is to create a model that acts as a Game Engine for Pacman, that takes in a sequence of frames and actions and outputs the next frame. 

### Initial Dataset
- HF Dataset: Tahahah/PacmanDataset_3
- Source columns: [episode, frame_image, action, next_frame_image, done]
- Using only: episode, frame_image and action
- Note: The actions are not one-hot encoded in the dataset. They need to be. There are 5 classes.

### Sequence Formation
- Initial batch shape for images: [batch_size, seq_length, 3, 512, 512]
- Initial batch shape for actions: [batch_size, seq_length, 5]
- 3 channels for RGB images


### VAE Encoding Options
- Option 1: Process seq_length images at once, then concatenate batch_size samples
- Option 2: Process all seq_length * batch_size images at once, then reshape
- Option 3: Process arbitrary batches < seq_length, then concatenate to form sequences


### VAE Output Dimensions

- Shape: [batch_size, seq_length*32, 16, 16]
- Using dc-ae-f32c32-in-1.0 VAE with:
    - 32x compression ratio
    - 32 latent channels
    - For seq_length = 32, 512x512 images

### Diffusion Model

-  Input shape: [batch_size, 1024, 16, 16]
-  Action conditioning: [batch_size, 32, 5]
-  Output shape: [batch_size, 32, 16, 16]


### Final VAE Decoding

- Output shape: [batch_size, 3, 512, 512]



### Pseudocode:
```
def process_batch(frames, actions, seq_length, batch_size):
    # Initial batch formation
    assert frames.shape == (batch_size, seq_length, 3, 512, 512)
    assert actions.shape == (batch_size, seq_length, 5)

    # VAE encoding (Option 2: all at once)
    frames_flat = frames.reshape(batch_size * seq_length, 3, 512, 512)
    latents = vae.encode(frames_flat)
    latents = latents.reshape(batch_size, seq_length * 32, 16, 16)

    # Diffusion model inference
    diffusion_output = diffusion_model(
        x=latents,
        condition=actions
    )  # Shape: [batch_size, 32, 16, 16]

    # VAE decoding
    final_output = vae.decode(diffusion_output)  # Shape: [batch_size, 3, 512, 512]
    
    return final_output
```