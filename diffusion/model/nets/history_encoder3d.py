import torch
import torch.nn as nn
from diffusion.model.builder import MODELS

class HistoryEncoder(nn.Module):
    def __init__(self, in_channels=3, seq_length=4, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        
        # Reshape input to [batch, 3, seq_length, h, w]
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
        self.conv2d = nn.Conv2d(hidden_dim * seq_length, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.seq_length, 3, h, w).permute(0, 2, 1, 3, 4)
        # Compute skip connection
        skip = self.skip_conv(x)
        # Main conv path with skip connection
        x = self.conv3d(x) + skip
        x = x.permute(0, 2, 1, 3, 4).reshape(b, -1, h, w)
        return self.conv2d(x)


@MODELS.register_module()
def build_history_encoder(**kwargs):
    return HistoryEncoder(**kwargs)