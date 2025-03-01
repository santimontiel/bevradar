from typing import Any, Dict

import torch.nn as nn
from omegaconf import OmegaConf

from bevradar.layers.image_encoders import SwinTransformerFPN, TimmEncoderFPN, TimmEncoderAttnFPN
from bevradar.layers.attention_unet import AttentionUNet
from bevradar.layers.segmentation_head import BEVSegmentationHead


class CameraSegmentationModel(nn.Module):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg

        self.image_encoder = TimmEncoderAttnFPN(cfg.image_encoder)

        self.decoder = AttentionUNet(
            in_channels=cfg.decoder.in_channels,
            encoder_channels=cfg.decoder.encoder_channels,
        )

        self.head = BEVSegmentationHead(
            classes=cfg.head.classes,
            in_channels=cfg.head.in_channels,
            grid_transform=cfg.head.grid_transform,
        )

        self.aux_head = BEVSegmentationHead(
            classes=cfg.head.classes,
            in_channels=cfg.decoder.in_channels,
            grid_transform=cfg.head.grid_transform,
        )


    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:

        camera_features = self.image_encoder(data)
        aux_logits = self.aux_head(camera_features)
        decoded_features = self.decoder(camera_features)
        logits = self.head(decoded_features)
        return {
            "logits": logits,
            "aux_logits": aux_logits,
        }