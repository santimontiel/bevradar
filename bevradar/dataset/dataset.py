from typing import Literal, Tuple

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

from bevradar.dataset.infos import (
    fill_trainval_infos,
    get_data_info,
    get_ann_info,
    get_samples_by_split
)
from bevradar.dataset.transforms import (
    LoadMultiViewImageFromFiles,
    ImageAug3D,
    ImageNormalize,
    LoadRadarPointsMultiSweeps,
    LoadPointsFromFile,
    LoadPointsFromMultiSweeps,
    GlobalRotScaleTrans,
    RadarPCDToOccImage,
    GetEgoPose,
    GetRelativeTransforms,
)
from bevradar.dataset.labels import LoadBEVSegmentation, LoadBEVMapSegmentation
from bevradar.utils.constants import SINGLE_NAME_TO_CLASS, SINGLE_CLASSES, MAP_CLASSES, CAMERA_NAMES



class NuScenesDataset(Dataset):

    def __init__(
        self,
        nusc: NuScenes,
        version: Literal["v1.0-trainval", "v1.0-test", "v1.0-mini"],
        split: Literal["train", "val"],
        use_camera: bool = True,
        use_lidar: bool = False,
        use_radar: bool = False,
        optimization_target: Literal["vehicle", "map", "multitask"] = "vehicle",
        x_bounds: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
        y_bounds: Tuple[float, float, float] = (-51.2, 51.2, 0.4),
        z_bounds: Tuple[float, float, float] = (-5.0, 3.0, 8.0),
        visibility_filter: bool = False,
        limit_pct_batches: float = 1.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        sequence_length: int = 3,
    ):

        self.nusc = nusc
        self.version = version
        self.split = split
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.optimization_target = optimization_target
        self.visibility_filter = visibility_filter
        self.sequence_length = sequence_length

        # Define pipeline.
        self.indices = get_samples_by_split(nusc, version, split, sequence_length)["indices"]
        self.pipeline = []
        if use_camera:
            self.pipeline.extend([
                LoadMultiViewImageFromFiles(to_float32=True),
                ImageAug3D(
                    final_dim=(320, 800),
                    # final_dim=(256, 704),
                    resize_lim=(0.47, 0.625),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=split == "train",
                    is_train=split == "train",),
                ImageNormalize(mean=mean, std=std),
            ])

        if use_lidar:
            self.pipeline.extend([
                LoadPointsFromFile(
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    shift_height=True,
                    use_color=False,
                ),
                LoadPointsFromMultiSweeps(
                    sweeps_num=10,
                    load_dim=5,
                    use_dim=5,
                    pad_empty_sweeps=True,
                    remove_close=True,
                ),
            ])
        if use_radar:
            self.pipeline.extend([
                LoadRadarPointsMultiSweeps(
                    load_dim=18,
                    sweeps_num=7,
                    use_dim=[0, 1, 2, 5, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
                    # use_dim=[0, 1, 2, 5, 8, 9],
                    max_num=2500,
                    compensate_velocity=True,
                    normalize=False,
                ),
            ])
            # self.pipeline += [
            #     LoadRadarDepth(
            #         nusc=self.nusc,
            #         camera_names=CAMERA_NAMES,
            #         depth_bounds=(0.0, 80.0, 0.1),
            #     )
            # ]

        self.pipeline += [
            GlobalRotScaleTrans(
                resize_lim=[0.9, 1.1],
                rot_lim=[-0.78539816, 0.78539816],
                trans_lim=0.5,
                is_train=split == "train",
                use_lidar=use_lidar,
                use_radar=use_radar,
            ),
        ]
        if self.use_radar:
            self.pipeline += [
                RadarPCDToOccImage(
                    x_bounds=(-50.0, 50.0, 0.5),
                    y_bounds=(-50.0, 50.0, 0.5),
                ),
            ]
        if optimization_target in ["vehicle", "multitask"]:
            self.pipeline += [
                LoadBEVSegmentation(
                    xbound=x_bounds,
                    ybound=y_bounds,
                    zbound=z_bounds,
                    classes=SINGLE_CLASSES,
                    filter_visibility=visibility_filter,
                ),
            ]
        if optimization_target in ["map", "multitask"]:
            self.pipeline += [
                LoadBEVMapSegmentation(
                    dataset_root=self.nusc.dataroot,
                    xbound=x_bounds,
                    ybound=y_bounds,
                    classes=MAP_CLASSES,
                ),
            ]
        self.pipeline += [
            GetEgoPose(nusc=self.nusc),
        ]

        self.end_pipeline = [
            GetRelativeTransforms(nusc=self.nusc, sequence_length=sequence_length),
        ]

        # Limit number of batches for debugging.
        if limit_pct_batches < 1.0:
            num_batches = int(limit_pct_batches * len(self.indices))
            if num_batches < 1:
                raise ValueError("Limiting number of batches to zero.")
            self.indices = self.indices[:int(limit_pct_batches * len(self.indices))]


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):

        # Get sample number from available indices.
        # idx = list(idx)[0]
        ixes = list(self.indices[idx])

        output_dict = {}

        for (i, idx) in enumerate(ixes):
            sample = self.nusc.sample[idx]

            # Get sample data and annotation information.
            info = fill_trainval_infos(self.nusc, sample, SINGLE_NAME_TO_CLASS)
            data = get_data_info(info)
            data = get_ann_info(info, data, SINGLE_CLASSES)
            data["visibility"] = info["visibility"]
            data["sample_rec"] = sample
            data["sample_token"] = sample["token"]

            # Apply transformations.
            for transform in self.pipeline:
                data = transform(data)

            # Collect data and return.
            return_dict = {}
            return_dict["images"] = torch.stack(data["img"])
            if self.use_lidar:
                return_dict["lidar_points"] = data["points"].tensor
            if self.use_radar:
                return_dict["radar_points"] = data["radar"].tensor
                return_dict["radar_occ_image"] = data["radar_occ_image"]
            return_dict["camera2ego"] = torch.from_numpy(np.array(data["camera2ego"]))
            return_dict["camera2lidar"] = torch.from_numpy(np.array(data["camera2lidar"]))
            return_dict["lidar2ego"] = torch.from_numpy(data["lidar2ego"])
            return_dict["lidar2camera"] = torch.from_numpy(np.array(data["lidar2camera"]))
            return_dict["lidar2image"] = torch.from_numpy(np.array(data["lidar2image"]))
            return_dict["camera_intrinsics"] = torch.from_numpy(np.array(data["camera_intrinsics"]))
            return_dict["img_aug_matrix"] = torch.from_numpy(np.array(data["img_aug_matrix"]))
            return_dict["lidar_aug_matrix"] = torch.from_numpy(data["lidar_aug_matrix"])
            if self.optimization_target in ["map", "multitask"]:
                return_dict["gt_map_bev"] = torch.from_numpy(data["gt_map_bev"])
            if self.optimization_target in ["vehicle", "multitask"]:
                return_dict["gt_segmentation_bev"] = torch.from_numpy(data["gt_segmentation_bev"])
                return_dict["gt_instance_bev"] = torch.from_numpy(data["gt_instance_bev"])
                return_dict["gt_visibility_mask"] = torch.from_numpy(data["gt_visibility_mask"])
            # return_dict["radar_depth"] = data["radar_depth"]
            # return_dict["sample_token"] = sample["token"]
            return_dict["ego_pose"] = torch.from_numpy(data["ego_pose"])

            if i == len(ixes) - 1:
                for k, v in return_dict.items():
                    output_dict[k] = v
            else:
                for k, v in return_dict.items():
                    output_dict[f"{k}_t{len(ixes) - i - 1}"] = v

        # Apply end pipeline.
        for transform in self.end_pipeline:
            output_dict = transform(output_dict)

        return output_dict


def nuscenes_dataset_collate_fn(batch):
    collated_batch = {}
    listable_keys = ["lidar_points", "radar_points", "sample_token"]
    for key in batch[0].keys():
        if any([key.startswith(k) for k in listable_keys]):
            collated_batch[key] = [sample[key] for sample in batch]
        else:
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
    return collated_batch