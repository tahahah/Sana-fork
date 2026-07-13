import torch
import torch.nn as nn
from diffusion.model.builder import MODELS


class HistoryEncoder(nn.Module):
    """Merges the precomputed VAE latents of the observation frames into a
    single conditioning latent — entirely in latent space (no VAE at runtime).

    Uses 3D convolutions to capture inter-frame temporal relationships.
    Input is [B, C*seq_length, h, w] (per-frame latents concatenated along
    channels) which is reshaped to [B, C, seq_length, h, w] for 3D conv
    processing. The temporal dimension is then collapsed back to channels and
    projected to a single [B, C, h, w] latent via 2D conv.

    Output is a raw latent (no sigmoid): it lives in the same TAESD latent space
    as the target latent it conditions, so bounding it to [0, 1] would be wrong.

    Args:
        in_channels: latent channels per frame (4 for TAESD)
        seq_length: number of observation frames (seq_length-1 from config)
        hidden_dim: internal channel width
    """

    def __init__(self, in_channels=4, seq_length=1, hidden_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length

        # 3D conv path: process [B, C, T, H, W] with temporal kernels
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
        )
        # 1x1x1 conv for skip connection
        self.skip_conv = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        # Collapse temporal dimension to channels, project to single frame
        self.conv2d = nn.Conv2d(hidden_dim * seq_length, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """Merge per-frame observation latents into a single conditioning latent.

        Args:
            x: [B, C*seq_length, h, w] precomputed obs latents concatenated along
               channels (C = in_channels per frame)
        Returns:
            [B, C, h, w] merged conditioning latent (raw, unbounded)
        """
        b, c, h, w = x.shape
        # Reshape [B, C*T, h, w] -> [B, T, C, h, w] -> [B, C, T, h, w]
        x = x.view(b, self.seq_length, self.in_channels, h, w).permute(0, 2, 1, 3, 4)
        # 3D conv with skip connection
        skip = self.skip_conv(x)
        x = self.conv3d(x) + skip
        # Collapse temporal: [B, hidden, T, h, w] -> [B, hidden*T, h, w]
        x = x.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)
        # Project to a single latent (raw output — same space as target latent)
        return self.conv2d(x)


@MODELS.register_module()
def build_history_encoder(**kwargs):
    return HistoryEncoder(**kwargs)
