from PIL import Image
from typing import Any, Dict
from pyquaternion import Quaternion

import numpy as np
import torch
import torchvision
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

from bevradar.core.points import BasePoints, LiDARPoints, RadarPoints


class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        results["resize"].append(resize)
        results["resize_dims"].append(resize_dims)
        results["crop"].append(crop)
        results["flip"].append(flip)
        results["rotate"].append(rotate)
        return resize, resize_dims, crop, flip, rotate


    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize                                              # (2, 2) * (1,)
        translation -= torch.Tensor(crop[:2])                           # (2,) - (2,)
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])                         # (2, 2)
            b = torch.Tensor([crop[2] - crop[0], 0])                    # (2,)
            rotation = A.matmul(rotation)                               # (2, 2) * (2,)
            translation = A.matmul(translation) + b                     # (2, 2) * (2,) + (2,)
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation
    
    @staticmethod
    def update_intrinsics_matrix(K, resize, crop, flip, rotate, orig_dims, final_dims):
        W_orig, H_orig = orig_dims
        W_final, H_final = final_dims
    
        # Apply resizing (scale focal lengths and principal points)
        K[0, 0] *= resize  # scale fx
        K[1, 1] *= resize  # scale fy
        K[0, 2] = (K[0, 2] * resize)  # scale cx
        K[1, 2] = (K[1, 2] * resize)  # scale cy
    
        # Apply cropping (shift principal point)
        crop_x, crop_y = crop[:2]
        K[0, 2] -= crop_x  # shift cx
        K[1, 2] -= crop_y  # shift cy
    
        # Apply flipping (if applicable)
        if flip:
            K[0, 2] = W_final - K[0, 2]  # update cx for flip
    
        # Apply rotation
        theta = rotate / 180.0 * np.pi
        R = torch.Tensor(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ]
        )
        K[:3, :3] = R.matmul(K[:3, :3])  # Apply rotation to 2D intrinsic matrix
    
        # Return the updated 4x4 intrinsic matrix
        return K


    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        data["resize"] = []
        data["resize_dims"] = []
        data["crop"] = []
        data["flip"] = []
        data["rotate"] = []

        imgs = data["img"]
        intrinsics = data["camera_intrinsics"]
        data["orig_img"] = imgs.copy()
        new_imgs = []
        transforms = []
        updated_intrinsics = []
        for i, img in enumerate(imgs):
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            updated_intrinsics.append(self.update_intrinsics_matrix(
                torch.from_numpy(intrinsics[i]).clone(),
                resize,
                crop,
                flip,
                rotate,
                data["ori_shape"],
                data["img_shape"],
            ))
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        data["augmented_intrinsics"] = updated_intrinsics
        return data


class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data
    

class ImageDenormalize:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        mean = torch.Tensor(self.mean).view(-1, 1, 1)
        std = torch.Tensor(self.std).view(-1, 1, 1)
        data["img"] = [(img * std + mean) for img in data["img"]]
        return data
    

class LoadRadarPointsMultiSweeps:
    """Load radar points from multiple sweeps.
    This is usually used for nuScenes dataset to utilize previous sweeps.
    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3, 
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 compensate_velocity=False, 
                 normalize_dims=[(3, 0, 50), (4, -100, 100), (5, -100, 100)], 
                 filtering='default', 
                 normalize=False, 
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.pc_range = pc_range
        self.compensate_velocity = compensate_velocity
        self.normalize_dims = normalize_dims
        self.filtering = filtering 
        self.normalize = normalize

        self.encoding = [
            (3, 'one-hot', 8), # dynprop
            (11, 'one-hot', 5), # ambig_state
            (14, 'one-hot', 18), # invalid_state
            (15, 'ordinal', 7), # pdh
            (0, 'nusc-filter', 1) # binary feature: 1 if nusc would have filtered it out
        ]


    def perform_encodings(self, points, encoding):
        for idx, encoding_type, encoding_dims in self.encoding:

            assert encoding_type in ['one-hot', 'ordinal', 'nusc-filter']

            feat = points[:, idx]

            if encoding_type == 'one-hot':
                encoding = np.zeros((points.shape[0], encoding_dims))
                encoding[np.arange(feat.shape[0]), np.rint(feat).astype(int)] = 1
            if encoding_type == 'ordinal':
                encoding = np.zeros((points.shape[0], encoding_dims))
                for i in range(encoding_dims):
                    encoding[:, i] = (np.rint(feat) > i).astype(int)
            if encoding_type == 'nusc-filter':
                encoding = np.zeros((points.shape[0], encoding_dims))
                mask1 = (points[:, 14] == 0)
                mask2 = (points[:, 3] < 7)
                mask3 = (points[:, 11] == 3)

                encoding[mask1 & mask2 & mask3, 0] = 1


            points = np.concatenate([points, encoding], axis=1)
        return points

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """

        invalid_states, dynprop_states, ambig_states = {
            'default': ([0], range(7), [3]), 
            'none': (range(18), range(8), range(5)), 
        }[self.filtering]

        radar_obj = RadarPointCloud.from_file(
            pts_filename, 
            invalid_states, dynprop_states, ambig_states
        )

        #[18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)
        

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)

            return points, masks
        
        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1), 
                        dtype=points.dtype)
            
            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]), 
                        dtype=points.dtype)
            masks = np.ones((num_points, 1), 
                        dtype=points.dtype)
            
            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def normalize_feats(self, points, normalize_dims):
        for dim, min, max in normalize_dims:
            points[:, dim] -= min 
            points[:, dim] /= (max-min)
        return points

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.
        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.
                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                if self.compensate_velocity:
                    points_sweep[:, :2] += velo_comp * time_diff

                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)

                # current format is x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms timestamp
                points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
        points = self.perform_encodings(points, self.encoding)
        points = points[:, self.use_dim]

        if self.normalize:
            points = self.normalize_feats(points, self.normalize_dims)
        
        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        
        results["radar"] = points
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
    

class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points = LiDARPoints(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, torch.Tensor):
            points_numpy = points.numpy()
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)

        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim]
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


class GlobalRotScaleTrans:
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train, use_lidar, use_radar):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train
        self.use_lidar = use_lidar
        self.use_radar = use_radar

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = np.random.uniform(*self.resize_lim)
            theta = np.random.uniform(*self.rot_lim)
            translation = np.array([np.random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data and self.use_lidar:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            if "radar" in data and self.use_radar:
                data["radar"].rotate(-theta)
                data["radar"].translate(translation)
                data["radar"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data


class RadarPCDToOccImage:
    def __init__(self, x_bounds, y_bounds):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        radar_pcd = data["radar"].tensor.numpy()
        occ_image = radar_pcd_to_occ_image(radar_pcd, self.x_bounds, self.y_bounds)
        data["radar_occ_image"] = occ_image
        return data


def radar_pcd_to_occ_image(
    radar_pcd: np.ndarray,
    x_bounds: tuple[float, float, float] = (-51.2, 51.2, 0.8),
    y_bounds: tuple[float, float, float] = (-51.2, 51.2, 0.8),
) -> np.ndarray:
    """Convert radar point cloud to an occupancy image (binary).

    Args:
        radar_pcd: Radar point cloud in the shape (N, F) where N is the
            number of points and Fs is the number of features.s
        x_bounds: Tuple of (min, max, resolution) for x-axis.
        y_bounds: Tuple of (min, max, resolution) for y-axis.
    Returns:
        Occupancy image in the shape (1, x_cells, y_cells).
    """
    num_features = radar_pcd.shape[1] - 3
    x_cells = int((x_bounds[1] - x_bounds[0]) / x_bounds[2])
    y_cells = int((y_bounds[1] - y_bounds[0]) / y_bounds[2])
    bev_image = np.zeros((num_features, x_cells, y_cells), dtype=np.float32)

    # Filter points outside the bounds.
    mask_x = np.logical_and(radar_pcd[:, 0] >= x_bounds[0], radar_pcd[:, 0] < x_bounds[1])
    mask_y = np.logical_and(radar_pcd[:, 1] >= y_bounds[0], radar_pcd[:, 1] < y_bounds[1])
    mask = np.logical_and(mask_x, mask_y)
    radar_pcd = radar_pcd[mask]

    for point in radar_pcd:
        x = int((point[0] - x_bounds[0]) / x_bounds[2])
        y = int((point[1] - y_bounds[0]) / y_bounds[2])

        bev_image[:, x, y] = point[3:]

    return torch.from_numpy(bev_image)


class GetEgoPose:
    def __init__(self, nusc):
        self.nusc = nusc

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        
        # Get sample data for LIDAR_TOP
        sample = self.nusc.get("sample", data["sample_token"])
        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = self.nusc.get("sample_data", lidar_token)
        
        # Get ego pose from sample data
        ego_pose_rec = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        
        # Extract translation and rotation
        translation = np.array(ego_pose_rec["translation"])
        rotation = Quaternion(ego_pose_rec["rotation"]).rotation_matrix
        
        # Create transformation matrix
        transform = np.eye(4)  # Start with 4x4 identity matrix
        transform[:3, :3] = rotation  # Set rotation block
        transform[:3, 3] = translation  # Set translation vector
        
        data["ego_pose"] = transform
        return data
    

class GetRelativeTransforms:
    def __init__(self, nusc: NuScenes, sequence_length: int) -> None:
        self.nusc = nusc
        self.sequence_length = sequence_length

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rel_transforms"] = []
        now = data["ego_pose"]
        for i in range(self.sequence_length, 0, -1):
            if i != 1:
                prev = data[f"ego_pose_t{i - 1}"]
            else:
                prev = data["ego_pose"]
            rel_transform = np.linalg.inv(prev) @ now.numpy()
            data["rel_transforms"].append(rel_transform)

        data["rel_transforms"] = torch.from_numpy(np.stack(data["rel_transforms"]))
        return data
