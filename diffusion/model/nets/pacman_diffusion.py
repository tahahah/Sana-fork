import torch
import torch.nn as nn
from diffusion.model.builder import MODELS
from diffusion.model.nets.history_encoder3d import build_history_encoder
from diffusion.model.nets.sana_multi_scale import SanaMS


class PacmanDiffusionModel(nn.Module):
    """Latent diffusion model for Pacman frame generation.

    Architecture (fully-precomputed latent diffusion — no VAE at runtime):
    - Target frames are precomputed to TAESD latents [B, 4, 32, 32] offline
    - Observation frames are ALSO precomputed to per-frame latents offline
    - history_encoder merges the obs latents → single obs_latent [B, 4, 32, 32]
    - Noise is added in latent space
    - SanaMS operates on concatenated [x_t, obs_latent] → [B, 8, 32, 32]
    - SanaMS predicts velocity in latent space [B, 4, 32, 32]
    - VAE decodes the final denoised latent to a pixel image only at validation

    The history_encoder now works entirely in latent space, so no VAE encode
    happens inside the training/sampling loop.
    """

    def __init__(
        self,
        input_size=8,
        in_channels=4,
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
        seq_length=2,
        **kwargs
    ):
        super().__init__()

        self.seq_length = seq_length
        # VAE is NOT stored in the model anymore — encode/decode happens outside
        # But keep a reference for convenience (not registered as submodule)
        self.vae = vae
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False

        # History encoder merges precomputed obs latents (4ch each) → [B, 4, h, w]
        self.history_encoder = build_history_encoder(
            in_channels=4,
            seq_length=seq_length - 1,
            hidden_dim=12
        )

        # SanaMS: input = concat([x_t (4ch), obs_latent (4ch)]) = 8 channels
        # Output = velocity for x_t only = 4 channels
        self.sana = SanaMS(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=8,        # x_t (4) + obs_latent (4) concatenated
            out_channels=4,       # predict velocity for x_t only
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

        # Residual (temporal) skip: project the clean observation latent directly
        # to the output so the transformer only has to learn the *change* between
        # frames instead of regenerating the (mostly static) maze every step.
        # Zero-initialized, so at step 0 the model is identical to the no-skip
        # baseline; the model then learns how much of the previous frame to copy
        # straight through, letting crisp static structure bypass the transformer.
        self.obs_residual = nn.Conv2d(4, 4, kernel_size=1)
        nn.init.zeros_(self.obs_residual.weight)
        nn.init.zeros_(self.obs_residual.bias)

    def encode_obs(self, obs_latents):
        """Merge precomputed observation latents through the history encoder.

        Args:
            obs_latents: [B, 4*(seq_length-1), h, w] precomputed obs latents,
                per-frame latents concatenated along channels
        Returns:
            obs_latent: [B, 4, h, w] merged conditioning latent (raw)
        """
        return self.history_encoder(obs_latents)

    def forward_with_dpmsolver(self, x, timestep, y, data_info, obs_latent=None, **kwargs):
        """DPM-Solver interface. obs_latent is precomputed by the caller.

        Args:
            x: noisy latent [B, 4, 8, 8]
            timestep: diffusion timestep
            y: action conditioning
            data_info: dict with img_hw, aspect_ratio
            obs_latent: precomputed obs latent [B, 4, 8, 8]
        """
        model_out = self.forward(x, timestep, y, data_info=data_info, obs_latent=obs_latent, **kwargs)
        return model_out.chunk(2, dim=1)[0] if self.sana.pred_sigma else model_out

    def forward(self, x, timestep, y, mask=None, data_info=None, obs_latent=None, **kwargs):
        """Forward pass in latent space.

        Args:
            x: noisy latent [B, 4, 8, 8]
            timestep: diffusion timesteps
            y: action conditioning [B, 1, 1, 5]
            mask: optional attention mask
            data_info: dict with img_hw, aspect_ratio
            obs_latent: precomputed clean obs latent [B, 4, 8, 8]
        Returns:
            velocity prediction [B, 4, 8, 8]
        """
        if obs_latent is None:
            raise ValueError("obs_latent must be provided for latent diffusion")

        # Concatenate noisy latent with obs latent along channels
        sana_input = torch.cat([x, obs_latent], dim=1)  # [B, 8, 8, 8]

        # Sana predicts velocity in latent space
        model_out = self.sana(sana_input, timestep, y, mask=mask, data_info=data_info, **kwargs)

        # Residual temporal skip: add a zero-initialized projection of the clean
        # obs latent to the velocity channels. The loss/parametrization is
        # unchanged (output is still compared to the velocity target); this just
        # gives the output a direct path to the spatially-aligned previous frame
        # so it can reuse static structure instead of regenerating it.
        residual = self.obs_residual(obs_latent)
        c = residual.shape[1]
        if model_out.shape[1] == c:
            model_out = model_out + residual
        else:
            # pred_sigma case: only add to the velocity/mean channels, leave sigma
            model_out = torch.cat([model_out[:, :c] + residual, model_out[:, c:]], dim=1)
        return model_out
