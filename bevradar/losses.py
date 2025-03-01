import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedChannelwiseBCELoss(nn.Module):
    def __init__(self, num_channels, reduction='mean'):
        """
        Args:
            num_channels: Number of channels in the input
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_channels = num_channels
        self.reduction = reduction
        
    def forward(self, predictions, targets, custom_weights=None):
        """
        Args:
            predictions: Tensor of shape (B, C, H, W) with logits
            targets: Tensor of shape (B, C, H, W) with binary labels
            custom_weights: Optional tensor of shape (C,) to override
                learned weights
            
        Returns:
            loss: Scalar tensor if reduction='mean'/'sum', or tensor of
                shape (B, C) if reduction='none'.
        """
        assert (
            predictions.shape == targets.shape
        ), f"Predictions shape {predictions.shape} must match targets shape {targets.shape}"
        
        # Calculate BCE loss for each spatial position
        pixel_losses = F.binary_cross_entropy_with_logits(
            predictions, 
            targets, 
            reduction='none'
        )
        
        # Average over spatial dimensions (H, W) to get per-channel loss
        channel_losses = pixel_losses.mean(dim=[2, 3])  # Shape: (B, C)
        
        # Apply channel weights
        weights = (
            custom_weights
            if custom_weights is not None
            else torch.ones(channel_losses.shape[1]).to(channel_losses.device)
        )
        weighted_losses = channel_losses * weights.view(1, -1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_losses.mean()
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        elif self.reduction == 'none':
            return weighted_losses
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def get_channel_weights(self):
        """Returns the normalized channel weights"""
        return F.softmax(self.channel_weights, dim=0)
    

class MultiBinaryBCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 3.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.losses = [
            nn.BCEWithLogitsLoss(reduction="none")
            for _ in range(self.num_classes)
        ]

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (B, C, H, W) with logits
            targets: Tensor of shape (B, C, H, W) with binary labels
        """
        assert (
            predictions.shape == targets.shape
        ), f"Predictions shape {predictions.shape} must match targets shape {targets.shape}"

        losses = [
            loss(predictions[:, i], targets[:, i])
            for i, loss in enumerate(self.losses)
        ]
        if self.reduction == "mean":
            return torch.mean(torch.stack(losses))
        elif self.reduction == "sum":
            return torch.sum(torch.stack(losses))
        elif self.reduction == "none":
            return torch.stack(losses)