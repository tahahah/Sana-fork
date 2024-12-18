import torch
import torch.nn as nn
from diffusion.model.builder import MODELS


class HistoryEncoder(nn.Module):
    """Encodes a sequence of raw RGB frames into a single frame representation.
    
    Takes input of shape [batch_size, seq_length*3, height, width]
    and outputs [batch_size, 3, height, width]
    """
    
    def __init__(self, in_channels=96, out_channels=3, hidden_dim=48):
        super().__init__()
        
        # Create a sequence of Conv layers to gradually reduce channels while preserving spatial dimensions
        self.conv_layers = nn.Sequential(
            # First layer: in_channels -> hidden_dim 
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            
            # Second layer: hidden_dim -> hidden_dim//2
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1), 
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            
            # Third layer: hidden_dim//2 -> hidden_dim//4
            nn.Conv2d(hidden_dim//2, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(),
            
            # Final layer: hidden_dim//4 -> out_channels (3 for RGB)
            nn.Conv2d(hidden_dim//4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):  
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):  
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length*3, height, width]
                Contains concatenated RGB frames
        
        Returns:
            Tensor of shape [batch_size, 3, height, width]
        """
        return self.conv_layers(x)


@MODELS.register_module()
def build_history_encoder(**kwargs):
    return HistoryEncoder(**kwargs)