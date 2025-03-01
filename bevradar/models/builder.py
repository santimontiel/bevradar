import torch.nn as nn
from omegaconf import OmegaConf

from bevradar.models.camera_only import CameraSegmentationModel
from bevradar.models.camera_radar import CameraRadarModel
from bevradar.models.radar_only import RadarSegmentationModel


def build_model_from_config(config: OmegaConf) -> nn.Module:

    match config.name:
        case "CameraRadarModel":
            return CameraRadarModel(config)
        case "CameraSegmentationModel":
            return CameraSegmentationModel(config)
        case "RadarSegmentationModel":
            return RadarSegmentationModel(config)
        case _:
            raise ValueError(
                f"Unknown model name: {config.name}. Valid options are: "
                "CameraRadarModel, CameraSegmentationModel, RadarSegmentationModel"
            )