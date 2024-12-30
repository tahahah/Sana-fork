# Gameplan B: Latent Space Loss Implementation

## Stage 1: Exploratory Analysis

### Current Flow
1. In `PacmanDiffusionModel.forward()`:
   - Input `x` is in pixel space
   - `x` is concatenated with `obs` in pixel space
   - Concatenated input goes through history_encoder
   - Result is encoded through VAE to get latents
   - Latents go through Sana model
   - Output is decoded back to pixel space and returned
   - Currently no way to access latent space output

2. In `training_losses()`:
   - Loss is computed between clean_images (x_start) and model output in pixel space
   - For MSE loss type, comparison happens in pixel space
   - No existing mechanism to compute loss in latent space

### Identified Areas Requiring Changes

1. **Primary Issue**: Loss computation needs to happen in latent space, but input handling needs to remain in pixel space
   - Location: `gaussian_diffusion.py:training_losses()`
   - Current behavior: Loss compares pixel-space values
   - Needed behavior: Loss compares latent-space values while maintaining pixel-space operations

2. **Forward Function Modifications Required**:
   - Location: `pacman_diffusion.py:forward()`
   - Current behavior: Only returns pixel space output
   - Needed behavior: Must provide option to return latent output before decoding
   - Changes needed:
     * Add parameter to control output space
     * Modify return structure to optionally skip decoding
     * Ensure VAE encoding still happens for both paths

3. **Training Script Integration**:
   - Location: `train_pacman.py`
   - Current behavior: Expects and handles pixel-space outputs only
   - Changes needed:
     * Update to handle latent space outputs
     * Modify loss computation flow
     * Ensure proper space conversion at correct points

## Stage 2: Proposed Solution

### Core Changes Required

1. **Modify PacmanDiffusionModel.forward()**:
   ```python
   def forward(self, x, timestep, y, mask=None, data_info=None, obs=None, return_latents=False, **kwargs):
       # Existing pixel space operations remain unchanged
       # ...
       if self.vae is not None:
           with torch.set_grad_enabled(True):
               encoded = self.vae.encode(processed)
               latent_output = self.sana(encoded, timestep, y, mask=mask, data_info=data_info, **kwargs)
               if return_latents:
                   return latent_output
               pixel_output = self.vae.decode(latent_output)
           return pixel_output
   ```

2. **Modify training_losses() Function**:
   - Add parameter for loss computation space
   - Handle both pixel and latent space inputs
   - Ensure proper encoding of clean images for latent space comparison
   - Key changes:
     ```python
     def training_losses(self, model, x_start, timestep, model_kwargs=None, noise=None, compute_in_latents=False):
         # Existing noise addition in pixel space
         if compute_in_latents:
             model_kwargs['return_latents'] = True
             # Ensure x_start is encoded for comparison
             x_start_latent = model.vae.encode(x_start)
         # ... rest of implementation
     ```

3. **Loss Computation Flow**:
   ```
   Input (pixel space) -> Add Noise (pixel space) -> 
   History Encoder -> VAE Encode -> 
   Sana Model -> 
   IF latent_loss:
       Compare with VAE-encoded clean image
   ELSE:
       Decode -> Compare in pixel space
   ```

### Implementation Strategy
1. **Minimal Changes Principle**:
   - Keep all existing pixel space operations
   - Add parallel latent space path
   - Maintain backward compatibility
   - Add clear documentation for new parameters

2. **Error Prevention**:
   - Add validation for input/output space matching
   - Ensure proper tensor shapes before comparisons
   - Validate VAE availability when needed
   - Add descriptive error messages

## Stage 3: Implementation Plan

### Step 1: Forward Function Updates
1. Add `return_latents` parameter
2. Modify return logic
3. Add input validation
4. Update docstring
5. Add tests for both output modes

### Step 2: Training Losses Updates
1. Add `compute_in_latents` parameter
2. Implement latent space comparison logic
3. Add proper VAE encoding of clean images
4. Update loss computation
5. Add validation checks
6. Add tests for both computation spaces

### Step 3: Training Script Integration
1. Update loss computation call
2. Add proper parameter passing
3. Update validation logic
4. Add monitoring for both spaces
5. Update logging

### Step 4: Testing Strategy
1. Unit Tests:
   - Test forward function in both modes
   - Test loss computation in both spaces
   - Test error handling
   
2. Integration Tests:
   - End-to-end training flow
   - Memory usage monitoring
   - Performance benchmarking
   
3. Validation Tests:
   - Compare loss values in both spaces
   - Verify gradient flow
   - Check training stability

### Safety Measures
1. Input Validation:
   - Check tensor shapes
   - Validate parameter combinations
   - Verify VAE availability
   
2. Error Handling:
   - Clear error messages
   - Proper exception hierarchy
   - Graceful fallbacks where possible
   
3. Performance Monitoring:
   - Memory usage tracking
   - Computation time comparison
   - GPU utilization monitoring

4. Documentation:
   - Clear parameter documentation
   - Usage examples
   - Warning notes where needed
