import torch
from omegaconf import OmegaConf

from bevradar.dataset import build_dataloaders
from bevradar.models import build_model_from_config
from bevradar.modules import SegmentationModule
from bevradar.utils.tooling import parse_args, set_seed
from bevradar.config import load_config


def inference(config: OmegaConf, sample_idx: int = 0) -> None:
    """Perform an inference run using the provided configuration over a
    selected sample of the validation dataset.
    """

    dataloaders_dict = build_dataloaders(
        version=config.dataset.version,
        data_root=config.dataset.path,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        use_camera=config.dataset.use_camera,
        use_lidar=config.dataset.use_lidar,
        use_radar=config.dataset.use_radar,
        optimization_target=config.dataset.optimization_target,
        limit_pct_batches=config.dataset.limit_pct_batches,
        x_bounds=config.dataset.x_bounds,
        y_bounds=config.dataset.y_bounds,
        z_bounds=config.dataset.z_bounds,
    )
    model = build_model_from_config(config.model)

    # Load the sample data and batch it.
    sample = dataloaders_dict["val_dataset"][sample_idx]
    for k, v in sample.items():
        if k not in ["lidar_points", "radar_points", "sample_token"]:
            sample[k] = v.unsqueeze(0).to(device=config.device)
        else:
            sample[k] = [v.to(device=config.device)]
        
    # Load a model.
    if config.checkpoint_path is not None:
        module = SegmentationModule.load_from_checkpoint(
            checkpoint_path=config.checkpoint_path,
            model=model,
            config=config,
            train_dataloader=None,
        )
    else:
        module = SegmentationModule(model=model, config=config, train_dataloader=None)
    module.eval().to(device=config.device)

    # Perform inference.
    with torch.no_grad():
        output = module(sample)

    print(f">>> Inference done over sample {sample_idx}!")
    for k, v in output.items():
        print(f"{k}: {v.shape}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    set_seed(config.random_seed)
    inference(config, sample_idx=args.index)