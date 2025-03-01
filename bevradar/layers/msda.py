import torch
import torch.nn as nn
from einops import rearrange
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention


class FeaturePyramidAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.blocks = nn.ModuleList([
            MultiScaleDeformableAttention(
                embed_dims=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=True,
            )
            for _ in range(num_blocks)
        ])


    def _get_reference_points(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / height
        ref_x = ref_x.reshape(-1)[None] / width
        reference_points = torch.stack((ref_x, ref_y), -1)
        reference_points = reference_points.repeat(batch_size, 1, 1).unsqueeze(2)  # noqa: (B, N, 1, 2)
        return reference_points
    

    def forward(self, feats: torch.Tensor) -> torch.Tensor:

        feats_flatten, ref_points, spatial_shapes = [], [], []
        for f in feats:
            feats_flatten.append(rearrange(f, 'b c h w -> b (h w) c'))
            ref_points.append(self._get_reference_points(f.shape[0], f.shape[2], f.shape[3], f.device))
            spatial_shapes.append((f.shape[2], f.shape[3]))
        feats_flatten = torch.cat(feats_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, device=feats[0].device, dtype=torch.long)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        ref_points = torch.cat(ref_points, dim=1)

        for block in self.blocks:
            feats_flatten = block(
                query=feats_flatten,
                value=feats_flatten,
                identity=None,
                query_pos=None,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        feats = []
        for i, (h, w) in enumerate(spatial_shapes):
            feats.append(
                rearrange(
                    feats_flatten[:, level_start_index[i] : level_start_index[i] + h * w],
                    "b (h w) c -> b c h w", h=h, w=w,
                )
            )

        return feats

        

