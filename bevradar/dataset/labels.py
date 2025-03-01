from typing import Any, Dict, Tuple

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS


class LoadBEVMapSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        # # Make labels exclusive.
        # exclusive = labels.copy()
        # for i in range(labels.shape[0] - 1, 0, -1):
        #     exclusive[i - 1] *= np.logical_not(exclusive[i])

        # all_zeros = ~np.any(labels, axis=0)  # Will be True where all channels are 0
        # argmax = np.argmax(labels, axis=0)
        # argmax = argmax + 1  # Make sure 0 is reserved for all zeros
        # argmax[all_zeros] = 0

        data["gt_map_bev"] = labels
        return data
    

class LoadBEVSegmentation:

    def __init__(
        self,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        filter_visibility: bool = False,
    ) -> None:
        
        # Prepare the grid configuration.
        self.dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        self.bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        self.nx = np.array([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]).astype(np.uint64)

        self.classes = classes
        self.filter_visibility = filter_visibility

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        CONST = 255.0  # Arbitrary const. Max visibility is 4. Min visibility is 1.
        gt_segmentation_bev = np.zeros((len(self.classes), *self.nx[:2]), dtype=np.uint8)
        gt_instance_bev = np.zeros((1, *self.nx[:2]), dtype=np.uint8)
        gt_visibility_bev = np.ones((1, *self.nx[:2]), dtype=np.uint8) * CONST

        # If there are no ground truth boxes, return the empty segmentation.
        if len(data["gt_bboxes_3d"]) == 0:
            data["gt_segmentation_bev"] = gt_segmentation_bev.astype(np.int64)
            data["gt_instance_bev"] = gt_instance_bev.astype(np.int64)
            data["gt_visibility_bev"] = gt_visibility_bev.astype(np.int64)
            data["gt_visibility_mask"] = gt_visibility_bev > 1
            return data        

        # Get the ground truth boxes.
        corners = data["gt_bboxes_3d"].corners[:, (0, 3, 4, 7), :2].numpy()
        labels = data["gt_labels_3d"]
        visibilities = data["visibility"]

        idx = 1
        for points, label, vis in zip(corners, labels, visibilities):
            if label != -1:
                points = np.round((points - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]).astype(np.int32)
                points = points[[0, 2, 3, 1]]
                points = points[:, [1, 0]]
                cv2.fillPoly(gt_segmentation_bev[label], [points], 1)
                cv2.fillPoly(gt_instance_bev[0], [points], idx)
                cv2.fillPoly(gt_visibility_bev[0], [points], int(vis))
                idx += 1

        data["gt_segmentation_bev"] = gt_segmentation_bev.astype(np.int64)
        data["gt_instance_bev"] = gt_instance_bev.astype(np.int64)
        data["gt_visibility_bev"] = gt_visibility_bev.astype(np.int64)
        data["gt_visibility_mask"] = gt_visibility_bev > 1

        if self.filter_visibility:
            data["gt_segmentation_bev"] = data["gt_segmentation_bev"] * data["gt_visibility_mask"]
            data["gt_instance_bev"] = data["gt_instance_bev"] * data["gt_visibility_mask"]
            data["gt_visibility_bev"] = data["gt_visibility_bev"] * data["gt_visibility_mask"]

        return data