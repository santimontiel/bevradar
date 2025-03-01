import os
from pyquaternion import Quaternion
from typing import Any, Dict, List

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits

from bevradar.core.bboxes import LiDARInstance3DBoxes, Box3DMode

def sliding_window(lst, window_size):
    if window_size > len(lst):
        return []
    return [lst[i:i + window_size] for i in range(len(lst) - window_size + 1)]


def get_samples_by_split(
    nusc: NuScenes,
    version: str,
    split: str,
    sequence_length: int = 1,
):
    assert version in ['v1.0-mini', 'v1.0-trainval']
    assert split in ['train', 'val']

    if version == 'v1.0-mini':
        split_name = splits.mini_train if split == 'train' else splits.mini_val
    elif version == 'v1.0-trainval':
        split_name = splits.train if split == 'train' else splits.val
    else:
        raise ValueError(f"Unknown version {version}")
    
    result = {
        "indices": [],
        "tokens": [],
    }

    # Pre-filter scenes to avoid repeated lookups
    scene_lookup = {
        scene['token']: scene['name']
        for scene in nusc.scene
        if scene['name'] in split_name
    }

    # Process windows more efficiently
    windows = sliding_window(nusc.sample, sequence_length)
    filtered_windows = []

    for i, window in enumerate(windows):
        first_sample = window[0]
        scene_token = first_sample['scene_token']
        
        # Check if all samples in the window belong to the same scene
        if scene_token in scene_lookup and all(
            sample['scene_token'] == scene_token for sample in window
        ):
            result["indices"].append(range(i, i + sequence_length))
            result["tokens"].extend([sample['token'] for sample in window])
            filtered_windows.append(window)

    return result

# def get_samples_by_split(nusc: NuScenes, version: str, split: str):

#     assert version in ['v1.0-mini', 'v1.0-trainval']
#     assert split in ['train', 'val']

#     if version == 'v1.0-mini':
#         split_name = splits.mini_train if split == 'train' else splits.mini_val
#     elif version == 'v1.0-trainval':
#         split_name = splits.train if split == 'train' else splits.val
#     else:
#         raise ValueError(f"Unknown version {version}")
    
#     result = {
#         "indices": [],
#         "tokens": [],
#     }

#     for idx, sample in enumerate(nusc.sample):

#         sample_token = sample['token']
#         scene_token = nusc.get('sample', sample_token)['scene_token']
#         scene_name = nusc.get('scene', scene_token)['name']

#         #

#         if scene_name in split_name:
#             result["indices"].append(idx)
#             result["tokens"].append(sample_token)

#     return result


def fill_trainval_infos(
    nusc,
    sample,
    class_names,
    test=False,
    max_sweeps=10,
    max_radar_sweeps=10,
):

    lidar_token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
    location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']

    info = {
        'lidar_path': lidar_path,
        'token': sample['token'],
        'sweeps': [],
        'cams': dict(),
        'radars': dict(), 
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sample['timestamp'],
        'prev_token': sample['prev'],
        'location': location,
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    camera_types = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    # obtain 5 radar's information per frame
    radar_names = [
        'RADAR_FRONT',
        'RADAR_FRONT_LEFT',
        'RADAR_FRONT_RIGHT',
        'RADAR_BACK_LEFT',
        'RADAR_BACK_RIGHT'
    ]

    for radar_name in radar_names:
        radar_token = sample['data'][radar_name]
        radar_rec = nusc.get('sample_data', radar_token)
        sweeps = []

        while len(sweeps) < max_radar_sweeps:
            if not radar_rec['prev'] == '':
                radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)

                radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, radar_name)
                sweeps.append(radar_info)
                radar_token = radar_rec['prev']
                radar_rec = nusc.get('sample_data', radar_token)
            else:
                radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)

                radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, radar_name)
                sweeps.append(radar_info)
        
        info['radars'].update({radar_name: sweeps})
    
    # obtain sweeps for a single key-frame
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec['prev'] == '':
            sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                        l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        else:
            break
    info['sweeps'] = sweeps

    # Obtain annotations
    if not test:
        annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
        valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)
        visibility = np.array([int(anno['visibility_token']) for anno in annotations])

        # convert velo from global to lidar
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        for i in range(len(names)):
            if names[i] in class_names:
                names[i] = class_names[names[i]]
            else:
                names[i] = 'ignore'
        names = np.array(names)
        # we need to convert rot to SECOND format.
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        assert len(gt_boxes) == len(
            annotations), f'{len(gt_boxes)}, {len(annotations)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_velocity'] = velocity.reshape(-1, 2)
        info['num_lidar_pts'] = np.array(
            [a['num_lidar_pts'] for a in annotations])
        info['num_radar_pts'] = np.array(
            [a['num_radar_pts'] for a in annotations])
        info['valid_flag'] = valid_flag
        info['visibility'] = visibility
    
    return info


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.
    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def get_data_info(info) -> Dict[str, Any]:

    data = {
        "token": info['token'],
        "sample_idx": info['token'],
        "lidar_path": info['lidar_path'],
        "sweeps": info['sweeps'],
        "timestamp": info['timestamp'],
        "location": info.get('location', None),
        "radar": info.get('radars', None),
    }

    if data['location'] is None:
        data.pop('location')
    if data['radar'] is None:
        data.pop('radar')

    # ego to global transform
    ego2global = np.eye(4).astype(np.float32)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = info["ego2global_translation"]
    data["ego2global"] = ego2global

    # lidar to ego transform
    lidar2ego = np.eye(4).astype(np.float32)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = info["lidar2ego_translation"]
    data["lidar2ego"] = lidar2ego

    data["image_paths"] = []
    data["lidar2camera"] = []
    data["lidar2image"] = []
    data["camera2ego"] = []
    data["camera_intrinsics"] = []
    data["camera2lidar"] = []

    for _, camera_info in info["cams"].items():
        data["image_paths"].append(camera_info["data_path"])

        # lidar to camera transform
        lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
        lidar2camera_t = (
            camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
        )
        lidar2camera_rt = np.eye(4).astype(np.float32)
        lidar2camera_rt[:3, :3] = lidar2camera_r.T
        lidar2camera_rt[3, :3] = -lidar2camera_t
        data["lidar2camera"].append(lidar2camera_rt.T)

        # camera intrinsics
        camera_intrinsics = np.eye(4).astype(np.float32)
        camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
        data["camera_intrinsics"].append(camera_intrinsics)

        # lidar to image transform
        lidar2image = camera_intrinsics @ lidar2camera_rt.T
        data["lidar2image"].append(lidar2image)

        # camera to ego transform
        camera2ego = np.eye(4).astype(np.float32)
        camera2ego[:3, :3] = Quaternion(
            camera_info["sensor2ego_rotation"]
        ).rotation_matrix
        camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
        data["camera2ego"].append(camera2ego)

        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
        data["camera2lidar"].append(camera2lidar)

    # import torch
    # torch.set_printoptions(sci_mode=False, precision=3)
    # import pdb; pdb.set_trace()

    # annos = self.get_ann_info(index)
    # data["ann_info"] = annos
    return data


def get_ann_info(
    info: Dict[str, Any],
    data: Dict[str, Any],
    class_names: List[str],
    use_valid_flag: bool = True,
    with_velocity: bool = True,
):
    """Get annotation info according to the given index.

    Args:
        index (int): Index of the annotation data to get.

    Returns:
        dict: Annotation information consists of the following keys:

            - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                3D ground truth bboxes
            - gt_labels_3d (np.ndarray): Labels of ground truths.
            - gt_names (list[str]): Class names of ground truths.
    """
    # filter out bbox containing no points
    if use_valid_flag:
        mask = info["valid_flag"]
    else:
        mask = info["num_lidar_pts"] > 0
    gt_bboxes_3d = info["gt_boxes"][mask]
    gt_names_3d = info["gt_names"][mask]
    gt_labels_3d = []
    for cat in gt_names_3d:
        if cat in class_names:
            gt_labels_3d.append(class_names.index(cat))
        else:
            gt_labels_3d.append(-1)
    gt_labels_3d = np.array(gt_labels_3d)

    if with_velocity:
        gt_velocity = info["gt_velocity"][mask]
        nan_mask = np.isnan(gt_velocity[:, 0])
        gt_velocity[nan_mask] = [0.0, 0.0]
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

    # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
    # the same as KITTI (0.5, 0.5, 0)
    # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
    ).convert_to(Box3DMode.LIDAR)

    anns_results = dict(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        gt_names=gt_names_3d,
    )
    data.update(anns_results)
    return data