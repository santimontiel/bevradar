from typing import Dict, Literal, Tuple, Union

from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset, DataLoader

from bevradar.dataset.dataset import NuScenesDataset, nuscenes_dataset_collate_fn


def build_dataloaders(
    version: str,
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    use_camera: bool = True,
    use_radar: bool = False,
    use_lidar: bool = False,
    optimization_target: Literal["vehicle", "map", "multitask"] = "vehicle",
    limit_pct_batches: float = 1.0,
    x_bounds: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
    y_bounds: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
    z_bounds: Tuple[float, float, float] = (-5.0, 3.0, 8.0),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    sequence_length: int = 3,
) -> Dict[str, Union[Dataset, DataLoader]]:

    # Instance NuScenes dataset.    
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

    # Create the train and validation datasets.
    train_dataset = NuScenesDataset(
        nusc, version, "train",
        use_camera=use_camera,
        use_radar=use_radar,
        use_lidar=use_lidar,
        optimization_target=optimization_target,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        limit_pct_batches=limit_pct_batches,
        mean=mean,
        std=std,
        sequence_length=sequence_length,
    )

    val_dataset = NuScenesDataset(
        nusc, version, "val",
        use_camera=use_camera,
        use_radar=use_radar,
        use_lidar=use_lidar,
        optimization_target=optimization_target,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        limit_pct_batches=limit_pct_batches,
        mean=mean,
        std=std,
        sequence_length=sequence_length,
    )

    # Create the train and validation dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=nuscenes_dataset_collate_fn,
        drop_last=True,
        # prefetch_factor=4,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=nuscenes_dataset_collate_fn,
        # prefetch_factor=4,
    )

    return {
        'nusc': nusc,
        'train_dataset': train_dataset,
        'train_dataloader': train_loader,
        'val_dataset': val_dataset,
        'val_dataloader': val_loader,
    }




