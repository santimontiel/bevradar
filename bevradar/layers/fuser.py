from typing import List

import torch
import torch.nn as nn

class ConvFuser(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.layers(x)
        return x