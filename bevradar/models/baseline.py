from typing import Any, Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import timm
from torch import Tensor
from omegaconf import OmegaConf
from einops import rearrange

from mmdet.models.backbones.swin import SwinTransformer
from cardiff.layers.msda import FeaturePyramidAttention
from cardiff.layers.generalized_lssfpn import GeneralizedLSSFPN
from cardiff.layers.vtransform import LSSTransform
from cardiff.layers.radar_encoder import RadarBevConvEncoder
from cardiff.layers.radar_transformer import RadarEncoder, NATransformer
from cardiff.layers.ptv3.model import PointTransformerV3, scatter_to_bev_grid
# from cardiff.layers.fuser import ConvFuser
from cardiff.layers.gated_fusion import GatingFusion
# from cardiff.layers.second import SECOND, SECONDFPN
from cardiff.layers.attention_unet import AttentionUNet
# from cardiff.layers.segnext import SegNeXt
from cardiff.layers.segmentation_head import BEVSegmentationHead


class BaselineModel(nn.Module):
    def __init__(self, cfg: OmegaConf) -> None:
        super().__init__()
        self.cfg = cfg

        # self.backbone = SwinTransformer(
        #     embed_dims=cfg.backbone.embed_dims,
        #     num_heads=cfg.backbone.num_heads,
        #     window_size=cfg.backbone.window_size,
        #     mlp_ratio=cfg.backbone.mlp_ratio,
        #     qkv_bias=cfg.backbone.qkv_bias,
        #     qk_scale=cfg.backbone.qk_scale,
        #     drop_rate=cfg.backbone.drop_rate,
        #     attn_drop_rate=cfg.backbone.attn_drop_rate,
        #     drop_path_rate=cfg.backbone.drop_path_rate,
        #     patch_norm=cfg.backbone.patch_norm,
        #     out_indices=cfg.backbone.out_indices,
        #     with_cp=cfg.backbone.with_cp,
        #     convert_weights=cfg.backbone.convert_weights,
        #     init_cfg=cfg.backbone.init_cfg,
        # )

        self.backbone = timm.create_model(
            model_name=cfg.backbone.backbone_name,
            pretrained=cfg.backbone.pretrained,
            features_only=True,
            out_indices=list(OmegaConf.to_container(cfg.backbone.out_indices)),
        )

        self.fpa = FeaturePyramidAttention(
            embed_dim=256,
            num_blocks=2,
            num_heads=8,
            num_levels=2,
            num_points=8,
        )

        self.neck = GeneralizedLSSFPN(
            in_channels=cfg.neck.in_channels,
            out_channels=cfg.neck.out_channels,
            start_level=cfg.neck.start_level,
            num_outs=cfg.neck.num_outs,
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

        # self.radar_encoder = RadarBevConvEncoder(
        #     num_features=cfg.radar_encoder.num_features,
        #     out_channels=cfg.radar_encoder.out_channels,
        # )

        # self.radar_encoder = RadarEncoder(
        #     num_features=cfg.radar_encoder.num_features,
        #     embed_dim=cfg.radar_encoder.out_channels,
        #     num_layers=cfg.radar_encoder.num_layers,
        #     num_heads=cfg.radar_encoder.num_heads,
        #     kernel_size=cfg.radar_encoder.kernel_size,
        # )

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
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.radar_bev_bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(256),
        )

        self.act_post_botteleneck = nn.GELU()

        # self.radar_transformer = NATransformer(
        #     num_layers=2,
        #     dim=256,
        #     num_heads=8,
        #     kernel_size=11,
        #     dilation=1,
        # )
        
        # self.fuser = ConvFuser(
        #     in_channels=cfg.fuser.in_channels,
        #     out_channels=cfg.fuser.out_channels,
        # )

        self.fuser = GatingFusion(
            in_channels=cfg.fuser.in_channels,
            out_channels=cfg.fuser.out_channels,
        )

        # self.decoder = SECOND(
        #     in_channels=cfg.decoder.in_channels,
        #     out_channels=OmegaConf.to_container(cfg.decoder.out_channels, resolve=True),
        #     layer_nums=cfg.decoder.layer_nums,
        #     layer_strides=cfg.decoder.layer_strides,
        #     norm_cfg=OmegaConf.to_container(cfg.decoder.norm_cfg),
        #     conv_cfg=OmegaConf.to_container(cfg.decoder.conv_cfg),
        # )

        # self.decoder_neck = SECONDFPN(
        #     in_channels=OmegaConf.to_container(cfg.decoder_neck.in_channels, resolve=True),
        #     out_channels=OmegaConf.to_container(cfg.decoder_neck.out_channels, resolve=True),
        #     upsample_strides=cfg.decoder_neck.upsample_strides,
        #     norm_cfg=OmegaConf.to_container(cfg.decoder_neck.norm_cfg),
        #     upsample_cfg=OmegaConf.to_container(cfg.decoder_neck.upsample_cfg),
        #     use_conv_for_no_stride=cfg.decoder_neck.use_conv_for_no_stride,
        # )

        self.decoder = AttentionUNet(
            in_channels=cfg.decoder.in_channels,
            encoder_channels=cfg.decoder.encoder_channels,
        )

        # self.decoder = smp.DeepLabV3Plus(
        #     encoder_name="tu-regnetz_b16",
        #     encoder_weights="imagenet",
        #     in_channels=cfg.decoder.in_channels,
        #     decoder_channels=cfg.head.in_channels,
        #     activation=None,
        # )
        # self.decoder.segmentation_head = nn.Identity()

        if self.cfg.task in ["multitask", "map"]:
            self.map_head = BEVSegmentationHead(
                classes=[0, 1, 2, 3, 4, 5],
                in_channels=cfg.head.in_channels,
                grid_transform=cfg.head.grid_transform,
            )
            self.aux_map_head = BEVSegmentationHead(
                classes=[0, 1, 2, 3, 4, 5],
                in_channels=cfg.head.in_channels,
                grid_transform=cfg.head.grid_transform,
            )

        if self.cfg.task in ["multitask", "vehicle"]:
            self.head = BEVSegmentationHead(
                classes=cfg.head.classes,
                in_channels=cfg.head.in_channels,
                grid_transform=cfg.head.grid_transform,
            )
            self.aux_head = BEVSegmentationHead(
                classes=cfg.head.classes,
                in_channels=cfg.head.in_channels,
                grid_transform=cfg.head.grid_transform,
            )

    def encode_features(self, data: Dict[str, Any]) -> Dict[str, Tensor]:

        B, N = data["images"].shape[:2]
        images = data["images"]
        # radar_occ_image = data["radar_occ_image"]

        # Extract features from the backbone.
        radar_points = data["radar_points"]
        offset = torch.tensor([i.shape[0] for i in radar_points]).cumsum(0)
        radar_points = torch.cat(radar_points, 0)
        radar_dict = {
            "feat": radar_points[:, 3:],
            "coord": radar_points[:, :3],
            "offset": offset.to(radar_points.device),
            "grid_size": 3.5,
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

        images = rearrange(images, "b n c h w -> (b n) c h w")
        features = self.backbone(images)
        features = self.neck(features)
        features = [rearrange(f, "(b n) c h w -> b n c h w", b=B) for f in features]
        features_lss = torch.zeros_like(features[0])
        for n in range(N):
            these_feats = [f[:, n, ...] for f in features]
            refined_feats = self.fpa(these_feats)
            features_lss[:, n, ...] = refined_feats[0]
        # features = [rearrange(f, "(b n) c h w -> b n c h w", b=B) for f in features]
    
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
        fused_features = self.fuser([camera_features, radar_features])

        return {
            "camera_features": camera_features,
            "radar_features": radar_features,
            "fused_features": fused_features["out"],
            "attention_weights": fused_features["attention_weights"],
        }
    
    def forward(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
        
        features = self.encode_features(data)
        camera_features = features["camera_features"]
        radar_features = features["radar_features"]
        fused_features = features["fused_features"]

        aux_logits = self.aux_head(fused_features) if self.cfg.task in ["multitask", "vehicle"] else None
        aux_map_logits = self.aux_map_head(fused_features) if self.cfg.task in ["multitask", "map"] else None
        features = self.decoder(fused_features)
        logits = self.head(features) if self.cfg.task in ["multitask", "vehicle"] else None
        map_logits = self.map_head(features) if self.cfg.task in ["multitask", "map"] else None

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "map_logits": map_logits,
            "aux_map_logits": aux_map_logits,
            "fused_features": fused_features,
            "features": features,
            "camera_features": camera_features,
            "radar_features": radar_features,
        }

