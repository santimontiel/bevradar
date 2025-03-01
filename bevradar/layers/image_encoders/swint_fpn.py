from typing import Any, Dict
import torch
import torch.nn as nn
from einops import rearrange

from mmdet.models.backbones.swin import SwinTransformer
from bevradar.layers.generalized_lssfpn import GeneralizedLSSFPN
from bevradar.layers.vtransform import LSSTransform

class SwinTransformerFPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = SwinTransformer(
            embed_dims=cfg.backbone.embed_dims,
            num_heads=cfg.backbone.num_heads,
            window_size=cfg.backbone.window_size,
            mlp_ratio=cfg.backbone.mlp_ratio,
            qkv_bias=cfg.backbone.qkv_bias,
            qk_scale=cfg.backbone.qk_scale,
            drop_rate=cfg.backbone.drop_rate,
            attn_drop_rate=cfg.backbone.attn_drop_rate,
            drop_path_rate=cfg.backbone.drop_path_rate,
            patch_norm=cfg.backbone.patch_norm,
            out_indices=cfg.backbone.out_indices,
            with_cp=cfg.backbone.with_cp,
            convert_weights=cfg.backbone.convert_weights,
            init_cfg=cfg.backbone.init_cfg,
        )

        self.neck = GeneralizedLSSFPN(
            in_channels=cfg.neck.in_channels,
            out_channels=cfg.neck.out_channels,
            start_level=cfg.neck.start_level,
            num_outs=cfg.neck.num_outs,
        )

        feature_size = [
            size // cfg.vtransform.downsample
            for size in cfg.vtransform.image_size
        ]
        self.vtransform = LSSTransform(
            in_channels=cfg.vtransform.in_channels,
            out_channels=cfg.vtransform.out_channels,
            image_size=cfg.vtransform.image_size,
            feature_size=feature_size,
            xbound=cfg.vtransform.xbound,
            ybound=cfg.vtransform.ybound,
            zbound=cfg.vtransform.zbound,
            dbound=cfg.vtransform.dbound,
            downsample=cfg.vtransform.downsample,
        )

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        B = data["images"].shape[0]
        images = data["images"]

        images = rearrange(images, "b n c h w -> (b n) c h w")
        x = self.backbone(images)
        x = self.neck(x)
        x = [rearrange(f, "(b n) c h w -> b n c h w", b=B) for f in x]

        x, _, _ = self.vtransform(
            img=x[0], points=None, radar=None,
            camera2ego=data["camera2ego"],
            lidar2ego=data["lidar2ego"],
            lidar2camera=None, lidar2image=None,
            camera_intrinsics=data["camera_intrinsics"],
            camera2lidar=data["camera2lidar"],
            img_aug_matrix=data["img_aug_matrix"],
            lidar_aug_matrix=data["lidar_aug_matrix"],
        )
        return x