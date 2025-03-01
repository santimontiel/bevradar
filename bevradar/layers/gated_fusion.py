from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class GatingFusion(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
                for i in range(len(in_channels))
            ]
        )
        self.attention_weights = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Sigmoid(),
                )
                for _ in range(len(in_channels))
            ]
        )
        self.se_layer = SELayer(out_channels)

    def forward(self, x: List[Tensor]) -> Tensor:
        features = []
        for i in range(len(self.in_channels)):
            features.append(self.feature_extractors[i](x[i]))
        fused_features = sum(features)
        weights = [
            self.attention_weights[i](fused_features)
            for i in range(len(self.in_channels))
        ]
        features = sum([features[i] * weights[i] for i in range(len(self.in_channels))])
        features = self.se_layer(features)
        return {
            "out": features,
            "attention_weights": None,
        }


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """Squeeze-and-Excitation Layer.
        
        Args:
            channel (int): Number of input channels
            reduction (int): Reduction ratio for the squeeze operation
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze operation
        y = self.avg_pool(x).view(b, c)
        # Excitation operation
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)

def main():

    f1 = torch.randn(8, 128, 64, 64)
    f2 = torch.randn(8, 256, 64, 64)

    model = GatingFusion(in_channels=[128, 256], out_channels=128)
    out = model([f1, f2])

    print(out.shape)


if __name__ == "__main__":
    main()