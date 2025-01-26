import torch
import torch.nn as nn
from diffusion.model.builder import MODELS


class HistoryEncoder(nn.Module):
    """Encodes a sequence of raw RGB frames into a single frame representation.
    
    Takes input of shape [batch_size, seq_length*3, height, width]
    and outputs [batch_size, 3, height, width]
    """
    
    def __init__(self, in_channels=96, out_channels=3, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        layers = []
        current_channels = in_channels

        for hidden_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(current_channels, hidden_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_channels = hidden_ch

        # Final layer to match output channels
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length*3, height, width]
                Contains concatenated RGB frames
        
        Returns:
            Tensor of shape [batch_size, 3, height, width]
        """
        # Ensure input is float32 during training for stability
        if self.training:
            x = x.float()
        return self.conv_layers(x)


@MODELS.register_module()
def build_history_encoder(**kwargs):
    return HistoryEncoder(**kwargs)