import torch
import torch.nn as nn
from diffusion.model.builder import MODELS
from diffusion.model.nets.history_encoder import build_history_encoder
from diffusion.model.nets.sana_multi_scale import SanaMS
from diffusion.model.builder import vae_encode, vae_decode


class PacmanDiffusionModel(nn.Module):
    """Wrapper class that combines HistoryEncoder with SanaMS for Pacman diffusion model."""
    
    def __init__(
        self,
        input_size=32,
        in_channels=160,
        patch_size=1,
        hidden_size=128,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        caption_channels=120,
        pe_interpolation=1.0,
        model_max_length=300,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        vae=None,  
        seq_length=32,  
        **kwargs
    ):
        super().__init__()
        
        # Store VAE model
        self.vae = vae
        if self.vae is not None:
            # Freeze VAE parameters
            for param in self.vae.parameters():
                param.requires_grad = False
        self.seq_length = seq_length
        
        # Create history encoder to process raw image frames
        self.history_encoder = build_history_encoder(
            in_channels=3 * seq_length,  
            out_channels=3,  
            hidden_dim=3 * seq_length // 2  
        )
        
        # Create Sana model for diffusion with latent input channels from VAE
        self.sana = SanaMS(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=32,  
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            drop_path=drop_path,
            caption_channels=caption_channels,
            pe_interpolation=pe_interpolation,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            y_norm=y_norm,
            norm_eps=norm_eps,
            attn_type=attn_type,
            ffn_type=ffn_type,
            use_pe=use_pe,
            y_norm_scale_factor=y_norm_scale_factor,
            patch_embed_kernel=patch_embed_kernel,
            mlp_acts=mlp_acts,
            linear_head_dim=linear_head_dim,
            cross_norm=cross_norm,
            **kwargs
        )
    
    def encode_history(self, x, obs):
        """Encode the history before adding noise.
        
        Args:
            x: Input tensor [B, C, H, W]
            obs: Observation tensor to concatenate with x
        Returns:
            Encoded tensor with same batch and spatial dimensions as x
        """
        combined = torch.cat([x, obs], dim=1)
        return self.history_encoder(combined)
    
    def forward_with_dpmsolver(self, x, timestep, y, data_info, obs=None, return_latents=False, **kwargs):
        """
        DPM solver doesn't need variance prediction.
        
        Args:
            x: Input tensor [B, C, H, W]
            timestep: Diffusion timesteps
            y: Conditioning tensor
            data_info: Data info dictionary
            obs: Observation tensor
            return_latents: If True, return output in latent space. Default: False
            **kwargs: Additional arguments
            
        Returns:
            Model output, either in latent or pixel space based on return_latents
        """
        model_out = self.forward(x, timestep, y, data_info=data_info, obs=obs, return_latents=return_latents, **kwargs)
        
        # If in latent space or no sigma prediction needed, return as is
        if return_latents or not self.sana.pred_sigma:
            return model_out
            
        # Otherwise chunk the pixel space output
        return model_out.chunk(2, dim=1)[0]

    def forward(self, x, timestep, y, mask=None, data_info=None, obs=None, return_latents=False, **kwargs):
        """
        Forward pass through both history encoder and Sana model.
        
        Args:
            x: Tensor of shape [batch_size, 3, height, width] - Noisy input frame x_t in pixel space
            timestep: Tensor of diffusion timesteps
            y: Conditioning tensor
            mask: Optional attention mask 
            data_info: Optional data info dict
            obs: Tensor of shape [batch_size, 3*(seq_length-1), height, width] - Raw observation frames
            return_latents: If True, return output in latent space before VAE decoding. Default: False
            **kwargs: Additional arguments
            
        Returns:
            If return_latents=False: Model output in pixel space [batch_size, 3, height, width]
            If return_latents=True: Model output in latent space [batch_size, latent_dim, latent_height, latent_width]
        
        Raises:
            ValueError: If obs is None or VAE model is not provided
            RuntimeError: If tensor shapes are incompatible
        """
        # Input validation
        if obs is None:
            raise ValueError("obs must be provided for history encoding")
        if self.vae is None:
            raise ValueError("VAE model must be provided")
            
        # Shape validation
        if x.dim() != 4 or obs.dim() != 4:
            raise RuntimeError(f"Expected 4D tensors, got x: {x.dim()}D, obs: {obs.dim()}D")
        if x.size(0) != obs.size(0):
            raise RuntimeError(f"Batch size mismatch: x: {x.size(0)}, obs: {obs.size(0)}")
        if x.size(2) != obs.size(2) or x.size(3) != obs.size(3):
            raise RuntimeError(f"Spatial dimensions mismatch: x: {x.size()[2:]}, obs: {obs.size()[2:]}")
        
        # 1. Concatenate noisy frame with observation frames in pixel space
        concat_input = torch.cat([obs, x], dim=1)  # [b, 3*seq_length, h, w]
        
        # 2. Process through history encoder to get single frame
        processed = self.history_encoder(concat_input)  # [b, 3, h, w]
        
        # 3. Encode through VAE to get latents, allowing gradients to flow
        with torch.set_grad_enabled(True):  # Ensure gradients flow through VAE
            encoded = self.vae.encode(processed)
            latent_output = self.sana(encoded, timestep, y, mask=mask, data_info=data_info, **kwargs)
            
            # Return latent output if requested
            if return_latents:
                return latent_output
                
            # Otherwise decode back to pixel space
            pixel_output = self.vae.decode(latent_output)
            return pixel_output


# @MODELS.register_module()
# def SanaMS_PACMAN_P1_D12(**kwargs):
#     """Factory function for PacmanDiffusionModel following Sana naming convention."""
#     return PacmanDiffusionModel(
#         depth=12,
#         hidden_size=128,
#         patch_size=1,
#         num_heads=16,
#         **kwargs
#     )
