from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, channels: List[int]) -> None:
        super(MLP, self).__init__()
        layers = []
        for i in range(len(channels)-1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ])
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    

class CSAFusion(nn.Module):
    """Channel and Spatial Attention Fusion module for multi-modal feature fusion."""
    
    def __init__(self, in_channels: List[int], channels: int) -> None:
        """
        Initialize the CSA-Fusion module.
        
        Args:
            in_channels: Number of input channels for each modality.
            channels: Number of channels used in the fusion process.
        """
        super(CSAFusion, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        
        # Mapping convolution for each modality.
        self.conv_radar = nn.Conv2d(in_channels[1], channels, kernel_size=3, padding=1)
        self.conv_img = nn.Conv2d(in_channels[0], channels, kernel_size=3, padding=1)

        # Initial feature mixing convolution.
        self.conv_init = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        
        # Channel attention components.
        self.gap_mlp = MLP([channels, channels//3, channels])
        self.gmp_mlp = MLP([channels, channels//3, channels])
        
        # Spatial attention components.
        self.spatial_conv = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.spatial_attention = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        
        # Final fusion convolution.
        self.conv_final = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)


    def forward(self, img_bev: torch.Tensor, radar_bev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CSA-Fusion module.
        
        Args:
            img_bev: Image BEV features of shape (batch_size, camera_channels, height, width).
            radar_bev: Radar BEV features of shape (batch_size, radar_channels, height, width).
            
        Returns:
            Fused features of shape (batch_size, channels, height, width).
        """
        # Apply mapping convolutions to match channel dimensions.
        radar_bev = self.conv_radar(radar_bev)
        img_bev = self.conv_img(img_bev)

        # Initial feature mixing of both modalities.
        fin_bev = self.conv_init(torch.cat([radar_bev, img_bev], dim=1))
        
        # Compute channel attention weights using both GAP and GMP.
        gap_weights = self.gap_mlp(F.adaptive_avg_pool2d(fin_bev, 1))
        gmp_weights = self.gmp_mlp(F.adaptive_max_pool2d(fin_bev, 1))
        channel_weights = torch.sigmoid(gap_weights + gmp_weights)
        
        # Apply channel attention to both modalities.
        fmid_radar = radar_bev * channel_weights
        fmid_img = img_bev * channel_weights
        
        # Mix channel-attentive features for spatial attention.
        fmid_bev = self.spatial_conv(torch.cat([fmid_radar, fmid_img], dim=1))
        
        # Compute spatial attention weights using max and mean pooling.
        max_feat = torch.max(fmid_bev, dim=1, keepdim=True)[0]
        mean_feat = torch.mean(fmid_bev, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_feat, mean_feat], dim=1)
        spatial_weights = torch.sigmoid(self.spatial_attention(spatial_feat))
        
        # Split and apply spatial attention weights to each modality.
        w_radar = spatial_weights[:, 0:1, :, :]
        w_img = spatial_weights[:, 1:2, :, :]
        fout_radar = fmid_radar * w_radar
        fout_img = fmid_img * w_img
        
        # Final fusion of spatially-attentive features.
        f_bev = self.conv_final(torch.cat([fout_radar, fout_img], dim=1))
        
        return f_bev
    

def test():
    # Example usage of the CSA-Fusion module.
    img_bev = torch.randn(2, 80, 64, 64)
    radar_bev = torch.randn(2, 256, 64, 64)

    fuser = CSAFusion(in_channels=[80, 256], channels=128)
    fused_features = fuser(img_bev, radar_bev)
    assert fused_features.shape == (2, 128, 64, 64)

    print("CSA-Fusion module test passed.")


if __name__ == "__main__":
    test()