from typing import Any, Dict
import timm
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from bevradar.layers.msda import FeaturePyramidAttention
from bevradar.layers.generalized_lssfpn import GeneralizedLSSFPN
from bevradar.layers.vtransform import LSSTransform


class TimmEncoderAttnFPN(nn.Module):

    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            model_name=cfg.backbone.backbone_name,
            pretrained=cfg.backbone.pretrained,
            features_only=True,
            out_indices=list(OmegaConf.to_container(cfg.backbone.out_indices)),
        )

        self.neck = GeneralizedLSSFPN(
            in_channels=cfg.neck.in_channels,
            out_channels=cfg.neck.out_channels,
            start_level=cfg.neck.start_level,
            num_outs=cfg.neck.num_outs,
        )

        self.fpa = FeaturePyramidAttention(
            embed_dim=cfg.fpa.embed_dim,
            num_blocks=cfg.fpa.num_blocks,
            num_heads=cfg.fpa.num_heads,
            num_levels=cfg.fpa.num_levels,
            num_points=cfg.fpa.num_points,
        )

        self.vtransform = LSSTransform(
            in_channels=cfg.vtransform.in_channels,
            out_channels=cfg.vtransform.out_channels,
            image_size=cfg.vtransform.image_size,
            feature_size=cfg.vtransform.feature_size,
            xbound=cfg.vtransform.xbound,
            ybound=cfg.vtransform.ybound,
            zbound=cfg.vtransform.zbound,
            dbound=cfg.vtransform.dbound,
            downsample=cfg.vtransform.downsample,
        )


    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
                
        B, N = data["images"].shape[:2]
        images = data["images"]

        images = rearrange(images, "b n c h w -> (b n) c h w")
        features = self.backbone(images)
        features = self.neck(features)
        features = [rearrange(f, "(b n) c h w -> b n c h w", b=B) for f in features]
        features_lss = torch.zeros_like(features[0])
        for n in range(N):
            these_feats = [f[:, n, ...] for f in features]
            refined_feats = self.fpa(these_feats)
            features_lss[:, n, ...] = refined_feats[0]
    
        camera_features, _, _ = self.vtransform(
            img=features_lss, points=None, radar=None,
            camera2ego=data["camera2ego"],
            lidar2ego=data["lidar2ego"],
            lidar2camera=None, lidar2image=None,
            camera_intrinsics=data["camera_intrinsics"],
            camera2lidar=data["camera2lidar"],
            img_aug_matrix=data["img_aug_matrix"],
            lidar_aug_matrix=data["lidar_aug_matrix"],
        )

        return camera_features