# Gameplan for Debugging and Optimizing the Diffusion Model

## Overview
The original model is designed as a text-to-image (t2i) model and is not inherently structured to accept action and observation inputs for predicting the next frame. However, we have adapted the text conditioning feature from the t2i model to condition the generation on actions. This adaptation may not fully support observations.

## Dataset Processing
- The dataset processes images and generates samples based on a specified sequence length.
- The output size of the samples is structured as follows:
  - `obs = [b, 32*seq_length, h, w]`

## Key Objectives
- We aim to concatenate the noisy next frame with the observations before encoding.
- The noise (`noise_t`) is added to the image in the function `train_diffusion.training_losses`:
  ```python
  model, clean_images, torch.randint(0, config.scheduler.train_sampling_steps, (clean_images.shape[0],), device=clean_images.device).long(), 
  model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, obs=obs)
  ```

## Current Issues
- The model encounters dimension mismatch errors during the validation phase, particularly when handling the scale shift table and tensor dimensions. This issue arises due to the model's expectations of input shapes not aligning with the actual shapes being passed during validation.

## Relevant Code Snippets
1. **Validation Function**: The `log_validation` function processes the validation data and generates initial noise. It is crucial to ensure that the shapes align correctly at each stage of processing.
   ```python
   @torch.inference_mode()
   def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
       torch.cuda.empty_cache()
       vis_sampler = config.scheduler.vis_sampler
       model = accelerator.unwrap_model(model).eval()
       hw = torch.tensor([[config.model.image_size, config.model.image_size]], dtype=torch.float, device=device).repeat(1, 1)
       ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
       
       # Create null action tensors for classifier-free guidance
       batch_size = 1
       seq_len = config.data.sequence_length
       null_action = torch.zeros(batch_size, 1, seq_len-1, 5, device=device)  # 5 is number of actions
       null_action_mask = torch.ones(batch_size, seq_len-1, device=device)

       null_action = null_action.unsqueeze(1)
       null_action_mask = null_action_mask.unsqueeze(1)
       # Create validation dataloader
       val_dataset = build_dataset(asdict(config.data), resolution=image_size, aspect_ratio_type=config.model.aspect_ratio_type, vae_downsample_rate=config.vae.vae_downsample_rate, vae=None)
       val_dataloader = torch.utils.data.DataLoader(
           val_dataset,
           batch_size=config.train.train_batch_size,
           shuffle=False,
           num_workers=config.train.num_workers,
           pin_memory=True
       )

       # Create sampling noise:
       logger.info("Running validation... ")
       image_logs = []

       # Initialize VAE if not provided
       if vae is None:
           vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
           # Set scaling factor from config
           if hasattr(vae, 'cfg') and vae.cfg.scaling_factor is None:
               vae.cfg.scaling_factor = config.vae.scale_factor

       def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
           latents = []
           current_image_logs = []
           
           # Get a batch of validation samples from the dataset
           batch = next(iter(val_dataloader))
           img = batch['img'].to(device)  # [B, S*C, H, W]
           obs = batch['obs'].to(device)
           actions = batch['y'].to(device)  # [B, S, A]
           action_masks = batch['y_mask'].to(device)  # [B, S]
           
           print(f"Debug shapes:")
           print(f"- actions shape before: {actions.shape}")
           print(f"- null_action shape: {null_action.shape}")
           
           # Reshape actions to match model expectations [B, 1, S, A]
           actions = actions.unsqueeze(1)  # Add the extra dimension
           
           print(f"- actions shape after: {actions.shape}")
           
           batch_size = img.shape[0]
           seq_len = img.shape[1] # seq_len*32
           
           # Generate initial noise if not provided
           z_pixel_space = init_z if init_z is not None else torch.randn_like(img)
           z = vae.encode(z_pixel_space).detach()
           print(f"Debug - initial z shape: {z.shape}")
           
           # Base model kwargs for the shape info and observations
           model_kwargs = dict(
               data_info={"img_hw": hw, "aspect_ratio": ar},
               mask=None,  # Use mask directly from dataset
               obs=obs,  # Pass observations to be concatenated with noise in the model
               x_pixel_space=z_pixel_space
           )

           if sampler == "dpm-solver":
               dpm_solver = DPMS(
                   model.sana.forward_with_dpmsolver,
                   condition=actions,  # Use actions as condition
                   uncondition=null_action,  # Use null action as uncondition
                   cfg_scale=4.5,  # Same scale as original code
                   model_kwargs=model_kwargs,
                   model_type="flow",
                   schedule="FLOW",
               )
               denoised = dpm_solver.sample(
                   z,
                   steps=20,
                   order=2,
                   skip_type="time_uniform_flow",
                   method="multistep",
                   flow_shift=config.scheduler.flow_shift,
               )
               print(f"Debug - denoised shape after dpm_solver: {denoised.shape if denoised is not None else None}")
           
           # ... other sampling methods

   ```

2. **Model Definition**: The `PacmanDiffusionModel` integrates the history encoder and the Sana model, and the forward methods must handle the concatenation of observations and noisy frames correctly.
   ```python
   class PacmanDiffusionModel(nn.Module):
       def __init__(self, ...):
           ...
           self.history_encoder = build_history_encoder(
               in_channels=3 * seq_length,  
               out_channels=3,  
               hidden_dim=3 * seq_length // 2  
           )
           self.sana = SanaMS(...)
       
       def forward(self, x, timestep, y, mask=None, data_info=None, obs=None, x_pixel_space=None, **kwargs):
           if obs is None:
               raise ValueError("obs must be provided for history encoding")
           if x_pixel_space is None:
               with torch.no_grad():
                   x_pixel_space = self.vae.decode(x)
           concat_input = torch.cat([obs, x_pixel_space], dim=1)  # [b, 3*seq_length, h, w]
           processed = self.history_encoder(concat_input)  # [b, 3, h, w]
           if self.vae is not None:
               with torch.no_grad():
                   encoded = self.vae.encode(processed).detach()  # [b, 32, h/32, w/32]
           return self.sana(encoded, timestep, y, mask=mask, data_info=data_info, **kwargs)
   ```

3. **Training Losses**: The `train_diffusion.training_losses` function is where the noise is added to the clean images, which needs careful handling of input shapes.
   ```python
   def training_losses(model, clean_images, ...):
       noise_t = ...  # Noise addition logic
       model_kwargs = dict(y=y, mask=y_mask, data_info=data_info, obs=obs)
       loss = model(..., model_kwargs=model_kwargs)
       return loss
   ```

## Next Steps
- Consolidate findings and gather additional insights from the PhD student.
- Explore potential fixes for the dimension mismatch errors and ensure that the model can handle both training and validation without errors.

---

## Additional Notes
- Ensure that all changes made to the model maintain consistency across training and validation workflows.
- Document any further findings or modifications to the model architecture as necessary to facilitate future debugging efforts.

The original model is meant to be a text to image (t2i) model and is not meant for action+obs as input to predict next frame. We've borrowed the text condition feature from the t2i model to condition the generation on actions however, support for obs may not be as fleshed out.
Currently the dataset processes the images and makes seq_length samples and then encodes all the frames before providing it. It also supports batching and so the output size of the samples is this:

obs = [b, 32*seq_length, h, w]

Now, what we want is for the noisy next frame to be concatenated to the obs before being encoded. The noise_t is added to the image in the function train_diffusion.training_losses(

                    model, clean_images, torch.randint(0, config.scheduler.train_sampling_steps, (clean_images.shape[0],), device=clean_images.device).long(), 

                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, obs=obs)

                ), where it automatically generates the x_t sample and passes it to the forward function of our model. This means that we cant get our hands on the noised x_t sample outside of the model forward() function. Therefore, we need to change the model forward function such that the concat, conv_in and encoding occurs within the forward function of our model. the input dim to the forward function should be:


[b, 3*seq_length, h, w]


the workflow should be something like:

concat([b, 3 * (seq_length-1), h, w], x_t, dim=1) -> conv_in (currently known as historyEncoder) -> [b, 3, h, w] -> vae.encoder -> [b, 32, h/32, w/32] -> DiT -> [b, 32, h/32, w/32] -> vae.decoder -> [b, 3, h, w] (output next frame)

Actionables:

• Look at the dataset file and the dataset test file diffusion/data/datasets/pacman_data.py and tests/test_pacman_dataset.py and make a todo list underneath this dot point with the actionables that need to be accomplished to switch to the new workflow.
- Successfully modified the dataset to output raw image frames while also keeping backwards compatibility of the old workflow, all dependent on the vae input parameter.
- The output of the dataset looks like:
`
        return {
            'obs': frames[:-C, : , :],  # [(seq_len-1)*C, H, W]
            'img': frames[-C:, :, :],   # [C, H, W] 
            'y': actions[:, :-1, :],  # [1, seq_len-1, 5]
            'y_mask': torch.ones(1, actions.shape[1] - 1),  # [1, seq_len-1]
            'data_info': {
                'episode': sequence[-1].get('episode', 0),
                'done': sequence[-1].get('done', False)
            }
        }
`

• Look at the diffusion/model/nets/sana_multi_scale.py file and make a todo list underneath this dot point with the actionables that need to be accomplished to switch to the new workflow from the old error prone workflow which was:
 [b, 3(seq_length-1), h, w] -> vae.encoder -> concat([b, 32  (seq_length-1), h/32, w/32], x_t, dim=1) -> history_encoder -> [b, 32, h/32, w/32] -> DiT -> [b, 32, h/32, w/32] -> vae.decoder -> [b, 3, h, w] (output next frame)


Lets first brainstorm different technical solutions that we can pursue in order to achieve this while keeping them minimally invasive for at least the model part.

New Workflow Plan:
The model needs to be restructured to handle image concatenation and encoding within the forward function. The new workflow will be:

1. Input: x_t = [b, 3, h, w] and obs = [b, 32*(seq_length-1), h, w] raw image frames
2. Inside model.forward():
   - Split input into history and x_t: [b, 3*(seq_length-1), h, w] and x_t
   - Concatenate them: concat([b, 3*(seq_length-1), h, w], x_t, dim=1)
   - Pass through conv_in (historyEncoder): -> [b, 3, h, w]
   - VAE encode: -> [b, 32, h/32, w/32]
   - DiT processing: -> [b, 32, h/32, w/32]
   - VAE decode: -> [b, 3, h, w] (final next frame prediction)

Key Changes Required:
1. Move the VAE encoding step inside the model's forward pass
2. Restructure the historyEncoder to handle the raw image input instead of encoded features
3. Ensure the training_losses function properly passes the raw image data to the model
4. Update the dataset to output raw image frames instead of pre-encoded features [DONE]

User's comments:
- Ensure that the historyEncoder exists and the architecture of it matches our desired functionality.
- Look at the training script and see where 'img', which is the next frame x_0, is noised. This noised input will be the primary input to the model.forward() and our obs will be in the kwargs as it did not exist in the old workflow. Ensure the new addition of obs as input is welcome appropriately by the forward function.
- The model.forward() needs to handle the concatenation and encoding of the raw image frames and the x_t sample.