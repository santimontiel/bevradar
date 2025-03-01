from typing import List

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from bevradar.layers.ptv3.model import PointTransformerV3, scatter_to_bev_grid


class PTv3ResConvEncoder(nn.Module):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg
        hidden_dim = cfg.hidden_dim
        out_channels = cfg.out_channels
        kernel_size = cfg.kernel_size
        self.grid_size = cfg.grid_size

        self.radar_encoder = PointTransformerV3(
            in_channels=42,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 8, 16),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(32, 64, 128, 128, 256),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(128, 64, 64, 64),
            dec_num_head=(4, 4, 4, 8),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=4,
            qkv_bias=True,
        )

        self.radar_bev_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel_size, stride=1, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.radar_bev_bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=kernel_size, stride=1, padding=3),
            nn.BatchNorm2d(out_channels),
        )

        self.act_post_botteleneck = nn.GELU()

    def forward(self, radar_points: List[torch.Tensor]) -> torch.Tensor:

        # Extract features from the backbone.
        offset = torch.tensor([i.shape[0] for i in radar_points]).cumsum(0)
        radar_points = torch.cat(radar_points, 0)
        radar_dict = {
            "feat": radar_points[:, 3:],
            "coord": radar_points[:, :3],
            "offset": offset.to(radar_points.device),
            "grid_size": self.grid_size,
        }
        radar_point_features = self.radar_encoder(radar_dict)

        # Scatter radar points to BEV grid.
        radar_features = []
        offset = [0] + radar_dict["offset"].tolist()
        for start, end in zip(offset[:-1], offset[1:]):
            grid = scatter_to_bev_grid(
                points=radar_point_features["coord"][start:end],
                features=radar_point_features["feat"][start:end],
                x_range=(-51.2, 51.2),
                y_range=(-51.2, 51.2),
                z_range=(-10.0, 10.0),
                resolution=(0.4, 0.4, 20.0)
            )
            radar_features.append(grid)
        radar_features_bev = torch.stack(radar_features)

        # Apply neighbourhood attention to radar features in BEV.
        radar_features = self.act_post_botteleneck(
            self.radar_bev_encoder(radar_features_bev)
            + self.radar_bev_bottleneck(radar_features_bev)
        )

        return radar_features