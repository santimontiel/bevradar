from pyquaternion import Quaternion

import torch
import torch.nn.functional as F
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


class LoadRadarDepth:
    def __init__(
        self,
        nusc: NuScenes,
        camera_names: list,
        depth_bounds: list,
    ) -> None:
        self.nusc = nusc
        self.camera_names = camera_names
        self.depth_bounds = depth_bounds

    def __call__(
        self,
        data: dict,
    ):
        sample_token = data["token"]
        sample_record = self.nusc.get("sample", sample_token)
        C, F = len(self.camera_names), 42
        output_range_view = torch.full(
            (C, F, 320, 800),
            fill_value=0.0,
            dtype=torch.float32
        )

        for i, camera in enumerate(self.camera_names):

            lidar_data_token = sample_record["data"]["LIDAR_TOP"]
            lidar_data_record = self.nusc.get("sample_data", lidar_data_token)
            # point_cloud, _ = LidarPointCloud.from_file_multisweep(
            #     nusc=self.nusc,
            #     sample_rec=sample_record,
            #     chan="LIDAR_TOP",
            #     ref_chan="LIDAR_TOP",
            #     nsweeps=1,
            #     min_distance=1.0,
            # )
            # lidar_points = point_cloud.points.T[:, :3]
            lidar_points = data["radar"].tensor[:, :3]
            lidar_features = data["radar"].tensor[:, 3:]

            # 1. Transform the LiDAR point cloud to the ego vehicle frame at timestamp of LiDAR data.
            cs_record = self.nusc.get("calibrated_sensor", lidar_data_record["calibrated_sensor_token"])
            lidar2ego_wrt_lidar = np.eye(4)
            lidar2ego_wrt_lidar[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
            lidar2ego_wrt_lidar[:3, 3] = np.array(cs_record["translation"])
            lidar2ego_wrt_lidar = torch.from_numpy(lidar2ego_wrt_lidar).float()
            
            lidar_points = lidar_points.float()
            lidar_points_hom = torch.cat([lidar_points, torch.ones(lidar_points.shape[0], 1)], dim=1)
            lidar_points_hom = (lidar2ego_wrt_lidar @ lidar_points_hom.T).T
            lidar_points = lidar_points_hom[:, :3]

            # 2. From ego vehicle frame to global frame.
            ego_pose_record = self.nusc.get("ego_pose", lidar_data_record["ego_pose_token"])
            ego_wrt_lidar2global = np.eye(4)
            ego_wrt_lidar2global[:3, :3] = Quaternion(ego_pose_record["rotation"]).rotation_matrix
            ego_wrt_lidar2global[:3, 3] = np.array(ego_pose_record["translation"])
            ego_wrt_lidar2global = torch.from_numpy(ego_wrt_lidar2global).float()

            lidar_points_hom = torch.cat([lidar_points, torch.ones(lidar_points.shape[0], 1)], dim=1)
            lidar_points_hom = (ego_wrt_lidar2global @ lidar_points_hom.T).T
            lidar_points = lidar_points_hom[:, :3]

            # 3. From global frame to ego vehicle frame at timestamp of image data.

            camera_data_token = sample_record["data"][camera]
            camera_data_record = self.nusc.get("sample_data", camera_data_token)
            ego_pose_record = self.nusc.get("ego_pose", camera_data_record["ego_pose_token"])
            ego_wrt_cam2global = np.eye(4)
            ego_wrt_cam2global[:3, :3] = Quaternion(ego_pose_record["rotation"]).rotation_matrix
            ego_wrt_cam2global[:3, 3] = np.array(ego_pose_record["translation"])
            global2ego_wrt_cam = np.linalg.inv(ego_wrt_cam2global)
            global2ego_wrt_cam = torch.from_numpy(global2ego_wrt_cam).float()

            lidar_points_hom = torch.cat([lidar_points, torch.ones(lidar_points.shape[0], 1)], dim=1)
            lidar_points_hom = (global2ego_wrt_cam @ lidar_points_hom.T).T
            lidar_points = lidar_points_hom[:, :3]

            # 4. From ego vehicle frame with respect to camera data to camera frame.
            cs_record = self.nusc.get("calibrated_sensor", camera_data_record["calibrated_sensor_token"])
            cam2ego_wrt_cam = np.eye(4)
            cam2ego_wrt_cam[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
            cam2ego_wrt_cam[:3, 3] = np.array(cs_record["translation"])
            ego_wrt_cam2cam = np.linalg.inv(cam2ego_wrt_cam)
            ego_wrt_cam2cam = torch.from_numpy(ego_wrt_cam2cam).float()

            lidar_points_hom = torch.cat([lidar_points, torch.ones(lidar_points.shape[0], 1)], dim=1)
            lidar_points_hom = (ego_wrt_cam2cam @ lidar_points_hom.T).T
            lidar_points = lidar_points_hom[:, :3]

            # 4.1. Filter points that are outside the camera view (keep z > 0).
            mask = lidar_points[:, 2] > 0
            lidar_points = lidar_points[mask]
            lidar_features = lidar_features[mask]
            lidar_points_cam = lidar_points.clone()
            lidar_features_cam = lidar_features.clone()
            
            # 5. Project the points to the depth map.
            camera_intrinsic_hom = np.eye(4)
            intrinsics = np.array(cs_record["camera_intrinsic"])
            intrinsics = torch.from_numpy(intrinsics).float()

            # 5.1. Update camera intrinsic matrix according to data augmentation.
            # 5.1.1. Scale.
            scale_factor = data["resize"][i]
            intrinsics[0, 0] *= scale_factor
            intrinsics[1, 1] *= scale_factor
            intrinsics[0, 2] *= scale_factor
            intrinsics[1, 2] *= scale_factor

            # 5.1.2. Crop.
            crop = data["crop"][i]
            intrinsics[0, 2] -= crop[0]
            intrinsics[1, 2] -= crop[1]

            # 5.1.3. Flip.
            flip = data["flip"][i]
            if flip:
                intrinsics[0, 0] *= -1

            # 5.1.4. Rotate.
            rotation = -data["rotate"][i] * np.pi / 180
            rotation = torch.tensor([[np.cos(rotation), -np.sin(rotation), 0],
                                    [np.sin(rotation), np.cos(rotation), 0],
                                    [0, 0, 1]]).float()
            intrinsics = rotation @ intrinsics

            camera_intrinsic_hom[:3, :3] = intrinsics
            camera_intrinsic_hom = torch.from_numpy(camera_intrinsic_hom).float()

            lidar_points_hom = torch.cat([lidar_points, torch.ones(lidar_points.shape[0], 1)], dim=1)
            lidar_points_hom = (camera_intrinsic_hom @ lidar_points_hom.T).T
            lidar_points = lidar_points_hom[:, :2] / lidar_points_hom[:, 2].unsqueeze(1)

            # 5.1. Filter points that are outside the camera view (keep x and y within image size).
            mask_x = torch.logical_and(lidar_points[:, 0] >= 0, lidar_points[:, 0] < 1600 - 1)
            mask_y = torch.logical_and(lidar_points[:, 1] >= 0, lidar_points[:, 1] < 900 - 1)
            mask = torch.logical_and(mask_x, mask_y)
            lidar_points = lidar_points[mask]
            lidar_features = lidar_features[mask]

            lidar_points_aug = lidar_points_cam.clone().float()
            lidar_points_at_cam_xyz = lidar_points_cam.clone().float()
            lidar_features_aug = lidar_features_cam.clone().float()
            lidar_features_at_cam_xyz = lidar_features_cam.clone().float()
            lidar_points_hom = torch.cat([lidar_points_aug, torch.ones(lidar_points_aug.shape[0], 1)], dim=1)
            lidar_points_hom = (camera_intrinsic_hom @ lidar_points_hom.T).T
            lidar_points_aug = lidar_points_hom[:, :2] / lidar_points_hom[:, 2].unsqueeze(1)

            mask_x = torch.logical_and(lidar_points_aug[:, 0] >= 0, lidar_points_aug[:, 0] < 800 - 1)
            mask_y = torch.logical_and(lidar_points_aug[:, 1] >= 0, lidar_points_aug[:, 1] < 320 - 1)
            mask = torch.logical_and(mask_x, mask_y)
            lidar_points_aug = lidar_points_aug[mask]
            lidar_features_aug = lidar_features_aug[mask]
            lidar_points_at_cam_xyz = lidar_points_at_cam_xyz[mask]
            lidar_features_at_cam_xyz = lidar_features_at_cam_xyz[mask]

            # for j in range(lidar_features_at_cam_xyz.shape[1]):
            #     canvas = torch.zeros(1, 1, 320, 800)
            #     canvas[
            #         0, 0,
            #         lidar_points_aug[:, 1].long(),
            #         lidar_points_aug[:, 0].long()
            #     ] = lidar_features_at_cam_xyz[:, j]
            #     output_range_view[i, j] = canvas.squeeze()

            for j in range(lidar_features_at_cam_xyz.shape[1]):
                canvas = torch.zeros(1, 1, 320, 800)
                
                # Get unique x coordinates (columns)
                unique_x = torch.unique(lidar_points_aug[:, 0].long())
                
                for x in unique_x:
                    # Find all points that belong to this column
                    col_mask = lidar_points_aug[:, 0].long() == x
                    y_coords = lidar_points_aug[col_mask, 1].long()
                    
                    # Sort y coordinates and features for this column
                    sorted_indices = torch.argsort(y_coords)
                    y_coords = y_coords[sorted_indices]
                    features = lidar_features_at_cam_xyz[col_mask, j][sorted_indices]
                    
                    # Fill the entire column from min to max y
                    if len(y_coords) > 0:
                        y_min, y_max = y_coords.min(), y_coords.max()
                        canvas[0, 0, y_min:y_max+1, x] = features[0]  # Using first feature value for the column
                        
                output_range_view[i, j] = canvas.squeeze()

        # import matplotlib.pyplot as plt

        # # For all cameras, paint white the points that are non-zero in all cameras.
        # for c in range(C):
        #     vis_tensor = output_range_view[c].clone()
            
        #     # Reduce to 1 dim by taking the max value.
        #     vis_tensor = vis_tensor.max(dim=0)[0]
        #     mask = vis_tensor != 0
        #     vis_tensor[mask] = 1.0

        #     plt.figure(figsize=(8, 4))
        #     plt.imshow(vis_tensor.numpy(), cmap="gray")
        #     plt.tight_layout()
        #     plt.grid(False)
        #     plt.axis("off")
        #     plt.savefig(f"radar_depth_{c}.png")
        #     plt.close()

        data["radar_depth"] = output_range_view
        return data
    

