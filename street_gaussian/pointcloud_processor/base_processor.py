import os
import numpy as np
import open3d as o3d
import torch
from street_gaussian.config import cfg
from street_gaussian.datasets.base_readers import storePly
from street_gaussian.utils.colmap_utils import read_points3D_binary
from street_gaussian.datasets.base_readers import get_Sphere_Norm
from street_gaussian.utils.camera_utils import Camera
from typing import List
from abc import abstractmethod, ABC
from data_processor.waymo_processor.waymo_helpers import image_filename_to_cam, image_filename_to_frame
from easyvolcap.utils.console_utils import *

class BasePointCloudProcessor(ABC):
    @abstractmethod
    def __init__(self):
        self.datadir = cfg.source_path
        self.delta_frames = 10
        self.cams = cfg.data.cameras
        self.start_frame = cfg.data.selected_frames[0]
        self.end_frame = cfg.data.selected_frames[-1]

        # read lidar ply
        self.flip_axis = 1
        self.ply_dict = self.read_lidar_ply()

    @abstractmethod
    def read_lidar_ply(self):
        return dict()

    def make_lidar_ply(self, start_frame, end_frame, actor_ids):
        ply_frame_dict = dict()
        bkgd_ply = []
        for frame in range(start_frame, end_frame + 1):
            if frame not in self.ply_dict['background']:
                continue
            bkgd_ply.append(self.ply_dict['background'][frame])
        bkgd_ply = np.concatenate(bkgd_ply, axis=0)  # [N, xyz + rgb]
        ply_frame_dict['background'] = bkgd_ply

        for actor_id in actor_ids:
            if actor_id not in self.ply_dict:
                continue
            actor_ply = []
            for frame in range(start_frame, end_frame + 1):
                if frame not in self.ply_dict[actor_id]:
                    continue
                actor_ply.append(self.ply_dict[actor_id][frame])

            # empty actor ply
            if len(actor_ply) == 0:
                continue

            actor_ply = np.concatenate(actor_ply, axis=0)

            ply_frame_dict[actor_id] = actor_ply
        return ply_frame_dict

    def transform_lidar_ply(self, lidar_ply, pose):
        xyz, rgb = lidar_ply[..., :3], lidar_ply[..., 3:]
        xyz_homo = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz = (xyz_homo @ pose.T)[..., :3]
        lidar_ply = np.concatenate([xyz, rgb], axis=-1)
        return lidar_ply

    def initailize_ply(self, datadir, objects_info):
        actor_ids = []
        for actor in objects_info.values():
            actor_ids.append(actor['track_id'])
        ply_frame_dict = self.make_lidar_ply(self.start_frame, self.end_frame, actor_ids)

        input_ply_dir = os.path.join(datadir, 'input_ply')
        os.makedirs(input_ply_dir, exist_ok=True)

        ply_bkgd = ply_frame_dict.pop('background')
        ply_bkgd_visible = []
        for frame in range(self.start_frame, self.end_frame + 1):
            if frame not in self.ply_dict['background_visible']:
                continue
            ply_bkgd_visible.append(self.ply_dict['background_visible'][frame])
        ply_bkgd_visible = np.concatenate(ply_bkgd_visible, axis=0)
        ply_bkgd = ply_bkgd[ply_bkgd_visible]
        ply_lidar_xyz, ply_lidar_rgb = ply_bkgd[..., :3], ply_bkgd[..., 3:6]

        points_lidar = o3d.geometry.PointCloud()
        points_lidar.points = o3d.utility.Vector3dVector(ply_lidar_xyz)
        points_lidar.colors = o3d.utility.Vector3dVector(ply_lidar_rgb)
        downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.1)
        downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
        points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
        points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
        ply_lidar_path = os.path.join(input_ply_dir, 'points3D_lidar.ply')
        storePly(ply_lidar_path, points_lidar_xyz, points_lidar_rgb)

        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        self.sphere_center = lidar_sphere_normalization['center']
        self.sphere_radius = lidar_sphere_normalization['radius']

        if cfg.data.use_colmap:
            points_colmap_path = os.path.join(datadir, 'colmap', 'triangulated/sparse/model/points3D.bin')
            points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
            points_colmap_rgb = points_colmap_rgb / 255
            points_colmap_dist = np.linalg.norm(points_colmap_xyz - self.sphere_center, axis=-1)
            mask = points_colmap_dist < 2 * self.sphere_radius
            points_colmap_xyz_vis = points_colmap_xyz[mask]
            points_colmap_rgb_vis = points_colmap_rgb[mask]

            ply_colmap_path = os.path.join(input_ply_dir, 'points3D_colmap.ply')
            storePly(ply_colmap_path, points_colmap_xyz, points_colmap_rgb)

            points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0)
            points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)

            points_bkgd_xyz_vis = np.concatenate([points_lidar_xyz, points_colmap_xyz_vis], axis=0)
            points_bkgd_rgb_vis = np.concatenate([points_lidar_rgb, points_colmap_rgb_vis], axis=0)
            ply_bkgd_vis_path = os.path.join(input_ply_dir, 'points3D_bkgd_vis.ply')
            storePly(ply_bkgd_vis_path, points_bkgd_xyz_vis, points_bkgd_rgb_vis)
        else:
            points_bkgd_xyz = points_lidar_xyz
            points_bkgd_rgb = points_lidar_rgb

        ply_bkgd_path = os.path.join(input_ply_dir, 'points3D_bkgd.ply')
        storePly(ply_bkgd_path, points_bkgd_xyz, points_bkgd_rgb)

        for actor in objects_info.values():
            track_id = actor['track_id']
            object_id = actor['object_id']
            if track_id not in ply_frame_dict:
                continue
            ply_actor = ply_frame_dict[track_id]
            ply_actor_xyz, ply_actor_rgb = ply_actor[..., :3], ply_actor[..., 3:6]
            ply_actor_path = os.path.join(input_ply_dir, f'points3D_obj_{object_id:03d}.ply')

            storePly(ply_actor_path, ply_actor_xyz, ply_actor_rgb)

    def render_conditions(self, cameras: List[Camera], objects_info):
        for camera in tqdm(cameras, desc='Rendering LiDAR condition'):
            self.render_condition(camera, objects_info)

    @abstractmethod
    @torch.no_grad
    def render_condition(self, camera: Camera, objects_info):
        pass
