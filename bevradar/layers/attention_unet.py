from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
    

class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_fn,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
    

class AttentionBlock(nn.Module):
    def __init__(
        self,
        f_g: int,
        f_l: int,
        f_int: int,
    ) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        g_prime = self.W_g(g)
        x_prime = self.W_x(x)
        psi = F.relu(g_prime + x_prime)
        psi = self.psi(psi)
        return x * psi
    

class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_channels: list[int],
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.conv1 = ConvBlock(in_channels, encoder_channels[0])
        self.conv2 = ConvBlock(encoder_channels[0], encoder_channels[1])
        self.conv3 = ConvBlock(encoder_channels[1], encoder_channels[2])

        self.up3 = UpsampleBlock(encoder_channels[2], encoder_channels[1])
        self.attn3 = AttentionBlock(
            f_g=encoder_channels[1], f_l=encoder_channels[1], f_int=encoder_channels[1] // 2
        )
        self.upconv3 = ConvBlock(encoder_channels[1] * 2, encoder_channels[1])

        self.up2 = UpsampleBlock(encoder_channels[1], encoder_channels[0])
        self.attn2 = AttentionBlock(
            f_g=encoder_channels[0], f_l=encoder_channels[0], f_int=encoder_channels[0] // 2
        )
        self.upconv2 = ConvBlock(encoder_channels[0] * 2, encoder_channels[0])

        self.upconv1 = nn.Conv2d(encoder_channels[0], self.out_channels, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.conv3(F.max_pool2d(x2, kernel_size=2, stride=2))

        d3 = self.up3(x3)
        x2 = self.attn3(d3, x2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        x1 = self.attn2(d2, x1)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.upconv2(d2)

        return self.upconv1(d2)


def main():
    model = AttentionUNet(in_channels=256, encoder_channels=[384, 512, 768])
    x = torch.randn(1, 256, 256, 256)
    print(model(x).shape)


if __name__ == "__main__":
    main()