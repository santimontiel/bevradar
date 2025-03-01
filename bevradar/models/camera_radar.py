from typing import Any, Dict

import torch.nn as nn
from omegaconf import OmegaConf

from bevradar.layers.image_encoders import SwinTransformerFPN, TimmEncoderFPN, TimmEncoderAttnFPN
from bevradar.layers.radar_encoders import PTv3ResConvEncoder
from bevradar.layers.lxlv2_fuser import CSAFusion
from bevradar.layers.attention_unet import AttentionUNet
from bevradar.layers.segmentation_head import BEVSegmentationHead


class CameraRadarModel(nn.Module):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg

        self.image_encoder = TimmEncoderAttnFPN(cfg.image_encoder)
        self.radar_encoder = PTv3ResConvEncoder(cfg.radar_encoder)

        self.fuser = CSAFusion(
            in_channels=cfg.fuser.in_channels,
            channels=cfg.fuser.out_channels,
        )

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
        radar_features = self.radar_encoder(data["radar_points"])
        fused_features = self.fuser(camera_features, radar_features)
        aux_logits = self.aux_head(fused_features)
        fused_features = self.decoder(fused_features)
        logits = self.head(fused_features)

        return {
            "logits": logits,
            "aux_logits": aux_logits,
        }