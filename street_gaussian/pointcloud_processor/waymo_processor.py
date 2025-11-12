import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from data_processor.utils.pcd_utils import BasicPointCloud, fetchPly
from data_processor.waymo_processor.waymo_helpers import load_ego_poses, load_calibration, load_track, \
    image_filename_to_cam, image_filename_to_frame, image_heights, image_widths
from street_gaussian.config import cfg
from street_gaussian.datasets.base_readers import storePly
from street_gaussian.utils.camera_utils import Camera
from street_gaussian.utils.graphics_utils import project_numpy
from street_gaussian.pointcloud_processor.base_processor import BasePointCloudProcessor
from street_gaussian.utils.graphics_utils import get_rays, sphere_intersection
from data_processor.utils.render_utils import render_pointcloud_diff_point_rasterization, render_pointcloud_pytorch3d
from easyvolcap.utils.console_utils import *


class WaymoPointCloudProcessor(BasePointCloudProcessor):
    def __init__(self):
        print('Initializing Waymo Point Cloud Processor')
        self.datadir = cfg.source_path
        self.delta_frames = cfg.data.delta_frames
        self.cams = cfg.data.cameras
        self.start_frame = cfg.data.selected_frames[0]
        self.end_frame = cfg.data.selected_frames[-1]

        # load camera poses
        self.ego_frame_poses, self.ego_cam_poses = load_ego_poses(self.datadir, camera_count=len(cfg.data.cameras))

        # load calibration
        self.extrinsics, self.intrinsics = load_calibration(self.datadir)

        # load tracks
        self.track_info, self.track_camera_visible, self.trajectory = load_track(self.datadir)

        # read lidar ply
        self.flip_axis = 1
        self.ply_dict = self.read_lidar_ply()


    def fuse_nearest_left_right_frames(self, ply_dict_background, ply_dict_background_visible, frame_id):
        left_frame, right_frame = frame_id, frame_id
        while left_frame not in ply_dict_background:
            left_frame -= 1
        while right_frame not in ply_dict_background:
            right_frame += 1
        left_ply = ply_dict_background[left_frame]
        right_ply = ply_dict_background[right_frame]

        left_visible = ply_dict_background_visible[left_frame]
        right_visible = ply_dict_background_visible[right_frame]
        return np.concatenate([left_ply, right_ply], axis=0), np.concatenate([left_visible, right_visible], axis=0)


    def read_lidar_ply(self):
        ply_dict = dict()
        load_interpolated = cfg.data.get('load_interpolated', False)
        if load_interpolated:
            lidar_dir = os.path.join(self.datadir, 'interpolated_lidar')
        else:
            lidar_dir = os.path.join(self.datadir, 'lidar')

        # read background ply (we only save the pointcloud which is visible to the input camera)
        lidar_background_dir = os.path.join(lidar_dir, 'background')
        ply_dict_background = dict()
        ply_dict_background_visible = dict()

        bkgd_ply_list = sorted([os.path.join(lidar_background_dir, f) for f in os.listdir(lidar_background_dir) if f.endswith('.ply') and f != 'full.ply'])
        bkgd_ply_list = [x for x in bkgd_ply_list if image_filename_to_frame(os.path.basename(x)) >= self.start_frame and image_filename_to_frame(os.path.basename(x)) <= self.end_frame]

        for i, bkgd_ply_path in enumerate(tqdm(bkgd_ply_list, desc='Reading Background ply')):
            bkgd_ply = fetchPly(bkgd_ply_path)
            frame = image_filename_to_frame(os.path.basename(bkgd_ply_path))

            mask = bkgd_ply.mask 
            xyz_vehicle = bkgd_ply.points[mask]
            xyz_vehicle_homo = np.concatenate([xyz_vehicle, np.ones_like(xyz_vehicle[..., :1])], axis=-1)
            xyz_world = xyz_vehicle_homo @ self.ego_frame_poses[frame].T
            xyz_world = xyz_world[..., :3]
            rgb = bkgd_ply.colors[mask]

            ply_dict_background[frame] = np.concatenate([xyz_world, rgb], axis=-1)
            
            visible_mask = np.zeros_like(xyz_vehicle[..., 0], dtype=bool)
            for cam in self.cams:
                K = self.intrinsics[cam]
                RT = np.linalg.inv(self.extrinsics[cam])
                _, visible_cam_mask = project_numpy(xyz_vehicle, K, RT, cfg.data.get('image_heights', image_heights)[cam], cfg.data.get('image_widths', image_widths)[cam])
                visible_mask = np.logical_or(visible_mask, visible_cam_mask)
            ply_dict_background_visible[frame] = visible_mask

        if load_interpolated:
            interpolated_mapping_file = os.path.join(self.datadir, 'interpolated_track', 'frame_mapping.json')
            with open(interpolated_mapping_file, "r") as f:
                interpolated_frame_mapping = json.load(f)
                for frame_id_str, mapping in interpolated_frame_mapping.items():
                    if not mapping['is_original']:
                        ply_dict_background[int(frame_id_str)], ply_dict_background_visible[int(frame_id_str)] = \
                            self.fuse_nearest_left_right_frames(ply_dict_background, ply_dict_background_visible, int(frame_id_str))

        ply_dict['background'] = ply_dict_background
        ply_dict['background_visible'] = ply_dict_background_visible

        print('Reading Actor ply')
        # read actor lidar
        lidar_actor_dir = os.path.join(lidar_dir, 'actor')
        for i, track_id in enumerate(tqdm(os.listdir(lidar_actor_dir), desc='Reading Actor ply')):
            ply_dict_actor = dict()

            lidar_actor_dir_ = os.path.join(lidar_actor_dir, track_id)
            actor_ply_list = sorted([os.path.join(lidar_actor_dir_, f) for f in os.listdir(lidar_actor_dir_)
                                     if f.endswith('.ply') and f != 'full.ply'])
            for actor_ply_path in actor_ply_list:
                frame = image_filename_to_frame(os.path.basename(actor_ply_path))
                actor_ply = fetchPly(actor_ply_path)
                mask = actor_ply.mask

                if mask.sum() == 0:
                    continue

                xyz = actor_ply.points[mask]
                rgb = actor_ply.colors[mask]

                if self.trajectory is not None and not self.trajectory[track_id]['deformable']:
                    num_pointcloud_1 = (xyz[:, self.flip_axis] > 0).sum()
                    num_pointcloud_2 = (xyz[:, self.flip_axis] < 0).sum()
                    if num_pointcloud_1 >= num_pointcloud_2:
                        xyz_part = xyz[xyz[:, self.flip_axis] > 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] > 0]
                    else:
                        xyz_part = xyz[xyz[:, self.flip_axis] < 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] < 0]
                    xyz_flip = xyz_part.copy()
                    xyz_flip[:, self.flip_axis] *= -1
                    rgb_flip = rgb_part.copy()
                    xyz = np.concatenate([xyz, xyz_flip], axis=0)
                    rgb = np.concatenate([rgb, rgb_flip], axis=0)

                ply_dict_actor[frame] = np.concatenate([xyz, rgb], axis=-1)

            ply_dict[track_id] = ply_dict_actor

        return ply_dict

    def check_file_path(self, file_path):
        frame = image_filename_to_frame(file_path)
        cam = image_filename_to_cam(file_path)
        if cam in self.cams and frame >= self.start_frame and frame <= self.end_frame:
            return True
        else:
            return False

    def initailize_ply(self, datadir, objects_info):
        super().initailize_ply(datadir, objects_info)
        input_ply_dir = os.path.join(datadir, 'input_ply')
        ply_sky_path = os.path.join(input_ply_dir, 'points3D_sky.ply')
        sky_mask_dir = os.path.join(cfg.source_path, 'sky_mask')
        if os.path.exists(sky_mask_dir) and not os.path.exists(ply_sky_path):
            points_xyz_sky_mask = []
            points_rgb_sky_mask = []
            background_sphere_points = 50000    
            background_sphere_distance = 2.5      
            num_samples = background_sphere_points // (len(self.cams) * (self.end_frame - self.start_frame + 1))
            sky_mask_paths = sorted([x for x in os.listdir(sky_mask_dir) if x.endswith(f'.{cfg.data.get("img_format", "png")}')])
            sky_mask_paths = [os.path.join(sky_mask_dir, x) for x in sky_mask_paths if self.check_file_path(x)]
            sky_mask_lists = [(cv2.imread(sky_mask_path)[..., 0] > 0).reshape(-1) for sky_mask_path in sky_mask_paths]
            #sky_pixel_all = np.sum(np.stack(sky_mask_lists, axis=0))

            for i, sky_mask_path in tqdm(enumerate(sky_mask_paths), total=len(sky_mask_paths), desc="Processing sky masks"):
                basename = os.path.basename(sky_mask_path)
                frame, cam = image_filename_to_frame(basename), image_filename_to_cam(basename)
                image_path = os.path.join(cfg.source_path, 'images', basename)
                image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
                H, W, _ = image.shape
                
                sky_mask = sky_mask_lists[i]
                sky_indices = np.argwhere(sky_mask == True)[..., 0]
                if len(sky_indices) == 0: 
                    continue
                elif len(sky_indices) > num_samples:
                    random_indices = np.random.choice(len(sky_indices), num_samples, replace=False)
                    sky_indices = sky_indices[random_indices]

                K = self.intrinsics[cam]
                c2w = self.ego_frame_poses[frame] @ self.extrinsics[cam]
                w2c = np.linalg.inv(c2w)
                R, T = w2c[:3, :3], w2c[:3, 3]
                rays_o, rays_d = get_rays(H, W, K, R, T)
                rays_o = rays_o.reshape(-1, 3)[sky_indices]
                rays_d = rays_d.reshape(-1, 3)[sky_indices]

                p_sphere = sphere_intersection(rays_o, rays_d, self.sphere_center, self.sphere_radius * background_sphere_distance)
                points_xyz_sky_mask.append(p_sphere)
            
                pixel_value = image.reshape(-1, 3)[sky_indices]
                points_rgb_sky_mask.append(pixel_value)

            points_xyz_sky_mask = np.concatenate(points_xyz_sky_mask, axis=0)
            points_rgb_sky_mask = np.concatenate(points_rgb_sky_mask, axis=0)    
            input_ply_dir = os.path.join(datadir, 'input_ply')
            ply_sky_path = os.path.join(input_ply_dir, 'points3D_sky.ply')
            storePly(ply_sky_path, points_xyz_sky_mask, points_rgb_sky_mask)

    @torch.no_grad
    def render_condition(self, camera: Camera, objects_info):
        rgb_save_path = camera.meta['guidance_rgb_path']
        mask_save_path = camera.meta['guidance_mask_path']

        if os.path.exists(rgb_save_path) and os.path.exists(mask_save_path) and not cfg.diffusion.force_render_condition:
            return

        c2w = camera.get_extrinsic().cpu().numpy()
        ixt = camera.get_intrinsic().cpu().numpy()
        h, w = camera.image_height, camera.image_width

        # set actor ids
        frame = camera.meta['frame']
        start_frame = max(self.start_frame, frame - self.delta_frames)
        end_frame = min(self.end_frame, frame + self.delta_frames)
        actor_ids = []
        for actor in objects_info.values():
            if frame >= actor['start_frame'] and frame <= actor['end_frame']:
                actor_ids.append(actor['track_id'])

        ply_frame_dict = self.make_lidar_ply(start_frame=start_frame, end_frame=end_frame, actor_ids=actor_ids)
        ply_frame = [ply_frame_dict.pop('background')]
        frame_id = camera.meta['frame']
        world_pose_in_ego = np.linalg.inv(self.ego_frame_poses[frame_id])
        tmp = camera.meta['ego_pose'] @ world_pose_in_ego
        ply_frame[0] = self.transform_lidar_ply(ply_frame[0], tmp)

        if cfg.diffusion.shuffle_actors:
            names = list(ply_frame_dict.keys())
            np.random.shuffle(names)
            ply_frame_dict = {name: ply_frame_dict[name] for name in names}

        track_info_frame = self.track_info[f'{frame:06d}']
        for actor_id, ply_actor_frame in ply_frame_dict.items():
            if actor_id not in track_info_frame:
                continue
            # camera_box = track_info_frame[actor_id]['camera_box']
            # lidar_box = track_info_frame[actor_id]['lidar_box']
            # box = camera_box if camera_box is not None else lidar_box
            box = track_info_frame[actor_id]['lidar_box']
            pose_vehicle = np.array([
                [np.cos(box['heading']), -np.sin(box['heading']), 0, box['center_x']],
                [np.sin(box['heading']), np.cos(box['heading']), 0, box['center_y']],
                [0, 0, 1, box['center_z']],
                [0, 0, 0, 1]
            ])
            pose_vehicle = camera.meta['ego_pose'] @ pose_vehicle
            ply_actor_frame = self.transform_lidar_ply(ply_actor_frame, pose_vehicle)
            ply_frame.append(ply_actor_frame)

        ply_frame = np.concatenate(ply_frame, axis=0)
        ply_xyz, ply_rgb = ply_frame[..., :3], ply_frame[..., 3:]
        ply_mask = np.ones_like(ply_xyz[..., 0]).astype(bool)
        ply_feature = np.concatenate([ply_rgb, ply_mask[:, None]], axis=-1)
        ply_render = render_pointcloud_diff_point_rasterization(
            c2w[None], ixt[None], ply_xyz[None], ply_feature[None], h, w,
            scale=cfg.render.scale,
            use_ndc_scale=cfg.render.use_ndc_scale,
            use_knn_scale=cfg.render.use_knn_scale,
        )     
        # ply_render = render_pointcloud_pytorch3d(c2w[None], ixt[None], ply_xyz[None], ply_feature[None], h, w)
        ply_render_result_rgb, ply_render_result_mask = ply_render[0, ..., :3], ply_render[0, ..., 3:4]
        ply_render_result_rgb = (ply_render_result_rgb.cpu().numpy() * 255).astype(np.uint8)
        ply_render_result_mask = (ply_render_result_mask.cpu().numpy() * 255).astype(np.uint8)
        save_dir = os.path.dirname(rgb_save_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(rgb_save_path, ply_render_result_rgb[..., [2, 1, 0]])
        cv2.imwrite(mask_save_path, ply_render_result_mask)
