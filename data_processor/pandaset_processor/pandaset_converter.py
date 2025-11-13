import json
import os
import argparse
# import open3d as o3d
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from pandaset import DataSet as PandaSet, geometry
from pandaset.sequence import Sequence
from scipy.spatial.transform import Rotation
import pickle
import sys
import cv2
sys.path.append(os.getcwd())
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utils.multiprocess_utils import track_parallel_progress
from utils.visualization_utils import color_mapper, dump_3d_bbox_on_image
from utils.img_utils import visualize_depth_numpy
from utils.pcd_utils import storePly, fetchPly
from utils.box_utils import bbox_to_corner3d, inbbox_points
from pandaset_helpers import PANDA_CAMERA2ID, PANDA_ID2CAMERA, PANDA_LABELS, PANDA_NONRIGID_DYNAMIC_CLASSES, PANDA_RIGID_DYNAMIC_CLASSES, PANDA_DYNAMIC_CLASSES


img_width, img_height = 1920,1080
EXTRINSICS_FILE_PATH = os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml")


def to_notr_label(pandaset_label):
    if pandaset_label in ['Bus', 'Car', 'Emergency Vehicle', 'Medium-sized Truck',
        'Other Vehicle - Construction Vehicle', 'Other Vehicle - Pedicab',
        'Other Vehicle - Uncommon', 'Personal Mobility Device', 'Pickup Truck',
        'Semi-truck', 'Train', 'Tram / Subway']:
        return 'vehicle'
    elif pandaset_label in ['Pedestrian', 'Pedestrian with Object']:
        return 'pedestrian'
    elif pandaset_label in ['Bicycle', 'Bicycle with Object', 'Motorcycle', 'Motorized Scooter']:
        return 'bicycle'
    elif pandaset_label in ['Signs', 'Construction Signs']:
        return 'sign'
    else:
        return 'misc'
    

def compute_3d_box_corners(dim, pose):
    l, w, h = dim
    x = np.array([-l, l, l, -l, -l, l, l, -l]) / 2
    y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    z = np.array([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    pts = np.stack([x, y, z, np.ones(8)], axis=0)
    pts = (pose @ pts)[:3, :].T
    return pts


def quaternion_yaw(q) -> float:
    """从四元数计算偏航角(绕Z轴的旋转角度)"""
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return yaw


class PandaSetProcessor(object):
    """Process PandaSet.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # PandaSet Provides 6 cameras and 2 lidars
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "front_camera",        # "xxx_0.jpg"
            "front_left_camera",   # "xxx_1.jpg"
            "front_right_camera",  # "xxx_2.jpg"
            "left_camera",         # "xxx_3.jpg"
            "right_camera",        # "xxx_4.jpg"
            "back_camera"          # "xxx_5.jpg"
        ]
        # 0: mechanical 360° LiDAR, 1: front-facing LiDAR, -1: All LiDARs
        self.lidar_list = [-1]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}"
        self.workers = int(workers)
        self.pandaset = PandaSet(load_dir)
        self.create_folder()
        self.camera_params = yaml.load(open(EXTRINSICS_FILE_PATH, "r"), Loader=yaml.FullLoader)

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_idx):
        """Convert action for single file.

        Args:
            scene_idx (str): Scene index.
        """
        scene_data = self.pandaset[scene_idx]
        scene_data.load()
        num_frames = sum(1 for _ in scene_data.timestamps)

        # save instances info
        ego_poses = self.save_poses(scene_data, scene_idx, num_frames)
        extrinsics, intrinsics = self.save_calibs(scene_data, scene_idx, ego_poses)
        self.save_cam_poses(scene_data, scene_idx, num_frames, extrinsics)
        instances_info = self.save_objects(scene_data, scene_idx, num_frames, ego_poses, extrinsics, intrinsics)
        instances_info, frame_instances = self.process_objects_for_lidar(instances_info, num_frames)

        self.save_timestamp(scene_data, scene_idx)
        for frame_idx in tqdm(range(num_frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True):
            if "images" in self.process_keys:
                self.save_image(scene_data, scene_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(scene_data, scene_idx, frame_idx, frame_instances, instances_info, folder_name='lidar')
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, ego_poses, extrinsics, intrinsics, class_valid='all')

        # make full ply
        if 'lidar' in self.process_keys:
            lidar_actor_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/actor"
            for instance in os.listdir(lidar_actor_dir):
                instance_dir = os.path.join(lidar_actor_dir, instance)
                ply_files = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir) if f.endswith('.ply')]
                ply_xyz, ply_rgb = [], []
                for ply_file in ply_files:
                    ply = fetchPly(ply_file)
                    mask = ply.mask
                    ply_xyz.append(ply.points[mask])
                    ply_rgb.append(ply.colors[mask])
                ply_xyz = np.concatenate(ply_xyz, axis=0)   
                ply_rgb = np.concatenate(ply_rgb, axis=0)
                ply_mask = np.ones((ply_xyz.shape[0])).astype(np.bool_)
                storePly(os.path.join(instance_dir, 'full.ply'), ply_xyz, ply_rgb, ply_mask[:, None])

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_timestamp(self, scene_data: Sequence, scene_idx):
        """Parse and save the timestamp data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
        """
        timestamps = dict()
        frame_timestamps = scene_data.timestamps
        timestamps['frame'] = {str(frame_idx).zfill(6): ts for frame_idx, ts in enumerate(frame_timestamps.data)}
        for idx, cam in enumerate(tqdm(self.cam_list, desc="Processing timestamps")):
            cam_timestamps = scene_data.camera[cam].timestamps
            timestamps[cam] = {str(frame_idx).zfill(6): ts for frame_idx, ts in enumerate(cam_timestamps)}

        with open(f"{self.save_dir}/{str(scene_idx).zfill(3)}/timestamps.json", "w") as f:
            json.dump(timestamps, f, indent=1)

    def get_lidar(self, scene_data: Sequence, frame_idx, lidar_idx):
        pc_world = scene_data.lidar[frame_idx].to_numpy()
        pc_world = pc_world[pc_world[:, -1] == lidar_idx]
        # index        x           y         z        i         t       d
        # 0       -75.131138  -79.331690  3.511804   7.0  1.557540e+09  0
        # 1      -112.588306 -118.666002  1.423499  31.0  1.557540e+09  0
        # - `i`: `float`: Reflection intensity in a range `[0,255]`
        # - `t`: `float`: Recorded timestamp for specific point
        # - `d`: `int`: Sensor ID. `0` -> mechnical 360° LiDAR, `1` -> forward-facing LiDAR
        lidar_poses = scene_data.lidar.poses[frame_idx]

        pcd_ego = geometry.lidar_points_to_ego(
            pc_world[:, :3], lidar_poses
        )
        return pc_world, pcd_ego

    def save_image(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        
        lidar_depth_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/depth"
        os.makedirs(lidar_depth_dir, exist_ok=True)
        
        for idx, cam in enumerate(self.cam_list):
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(6)}_{str(idx)}.jpg"
            )
            # write PIL Image to jpg
            image = scene_data.camera[cam][frame_idx]
            image.save(img_path)

                        
    def save_calibs(self, scene_data: Sequence, scene_idx, ego_poses):
        import pyquaternion
        def _pandaset_pose_to_matrix(pose):
            translation = np.array([pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]])
            quaternion = np.array([pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]])
            pose = np.eye(4)
            pose[:3, :3] = pyquaternion.Quaternion(quaternion).rotation_matrix
            pose[:3, 3] = translation
            return pose

        extrinsics, intrinsics = {}, {}
        first_frame_lidar_pose = scene_data.lidar.poses[0]
        ego_pose = geometry._heading_position_to_mat(first_frame_lidar_pose['heading'], first_frame_lidar_pose['position'])
        for cam_id, cam_name in enumerate(self.cam_list):
            camera = scene_data.camera[cam_name]
            poses = camera.poses[0]
            c2w = geometry._heading_position_to_mat(poses['heading'], poses['position'])
            ego_pose_inv = np.linalg.inv(ego_poses[0])
            extrinsic = ego_pose_inv @ c2w
            K = camera.intrinsics
            intrinsic = [K.fx, K.fy, K.cx, K.cy, 0.0, 0.0, 0.0, 0.0, 0.0]

            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                + f"{cam_id}.txt",
                extrinsic,
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                + f"{cam_id}.txt",
                intrinsic,
            )
            intrinsic = np.eye(3)
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = K.fx, K.fy, K.cx, K.cy
            extrinsics[cam_id] = extrinsic
            intrinsics[cam_id] = intrinsic
        return extrinsics, intrinsics

    def save_lidar(self, scene_data: Sequence, scene_idx, frame_idx, frame_instances, instances_info, folder_name='lidar'):
        lidar_background_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/background"
        lidar_actor_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/actor"
        lidar_depth_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/depth"
        os.makedirs(lidar_background_dir, exist_ok=True)
        os.makedirs(lidar_actor_dir, exist_ok=True)
        os.makedirs(lidar_depth_dir, exist_ok=True)
        
        current_instance_info = dict()
        for instance in frame_instances[frame_idx]:
            current_instance_info[instance] = dict()
            lidar_instance_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/{folder_name}/actor/{str(instance).zfill(6)}"
            os.makedirs(lidar_instance_dir, exist_ok=True)
            save_path = f"{lidar_instance_dir}/{str(frame_idx).zfill(6)}.ply"
            current_instance_info[instance]['save_path'] = save_path

            frame_annotations = instances_info[instance]['frame_annotations']
            idx = frame_annotations['frame_idx'].index(frame_idx)
            current_instance_info[instance]['obj_to_world'] = frame_annotations['obj_to_world'][idx]
            current_instance_info[instance]['box_size'] = frame_annotations['box_size'][idx]
            current_instance_info[instance]['class_name'] = instances_info[instance]['class_name']

        # index        x           y         z        i         t       d                                                     
        # 0       -75.131138  -79.331690  3.511804   7.0  1.557540e+09  0
        # 1      -112.588306 -118.666002  1.423499  31.0  1.557540e+09  0
        # - `i`: `float`: Reflection intensity in a range `[0,255]`
        # - `t`: `float`: Recorded timestamp for specific point
        # - `d`: `int`: Sensor ID. `0` -> mechnical 360° LiDAR, `1` -> forward-facing LiDAR
        
        # paint the lidar points
        lidar_idx = 1 if folder_name == 'lidar_forward' else 0
        pc_world, pc_ego = self.get_lidar(scene_data, frame_idx, lidar_idx)
        pcd_world, pcd_ego = pc_world[:, :3], pc_ego[:, :3]
        pcd_mask = np.zeros((pcd_world.shape[0])).astype(np.bool_)
        pcd_color = np.zeros((pcd_world.shape[0], 3)).astype(np.uint8)      
        
        # the lidar scans are synced such that the middle of a scan is at the same time as the front camera image
        timestamp = scene_data.camera["front_camera"].timestamps[frame_idx]

        for i, cam in enumerate(self.cam_list):
            camera = scene_data.camera[cam]
            cam_timestamps = np.array(scene_data.camera[cam].timestamps)
            cam_frame_idx = np.argmin(np.abs(cam_timestamps - timestamp)).astype(np.int32)
            cam_frame_idx = frame_idx
            points2d_camera, points3d_camera, inliner_indices_arr = geometry.projection(
                lidar_points=pcd_world,                
                camera_data=camera[cam_frame_idx], # type: ignore
                camera_pose=camera.poses[cam_frame_idx],
                camera_intrinsics=camera.intrinsics,
                filter_outliers=True
            )
            
            image = np.asarray(camera[cam_frame_idx]) # type: ignore            
            h, w = image.shape[:2]
            u_depth, v_depth = points2d_camera[:, 0], points2d_camera[:, 1]
            u_depth = np.clip(u_depth, 0, w-1).astype(np.int32)
            v_depth = np.clip(v_depth, 0, h-1).astype(np.int32)
            color_value = image[v_depth, u_depth]
            
            # ignore the points that have been painted         
            paint_mask = np.logical_not(pcd_mask[inliner_indices_arr])
            paint_inlinear_indices_arr = inliner_indices_arr[paint_mask]
            paint_color = color_value[paint_mask]
            pcd_color[paint_inlinear_indices_arr] = paint_color
            pcd_mask[paint_inlinear_indices_arr] = True
            
            # save lidar depth
            depth_value = points3d_camera[:, 2]
            depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
            indices = v_depth * w + u_depth
            np.minimum.at(depth, indices, depth_value)
            depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
            valid_depth_pixel = (depth != 0)
            valid_depth_value = depth[valid_depth_pixel]
            valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
            
            depth_filename = f"{lidar_depth_dir}/{str(frame_idx).zfill(6)}_{str(i)}.npz"
            depth_vis_filename = f"{lidar_depth_dir}/{str(frame_idx).zfill(6)}_{str(i)}.jpg"
            np.savez_compressed(depth_filename, mask=valid_depth_pixel, value=valid_depth_value)

            if i == 0:
                depth = depth.reshape(h, w).astype(np.float32)
                depth_vis, _ = visualize_depth_numpy(depth)
                depth_on_img = np.asarray(image)[..., [2, 1, 0]]
                depth_on_img[depth > 0] = depth_vis[depth > 0]
                cv2.imwrite(depth_vis_filename, depth_on_img)      

        pcd_instance_mask = np.zeros((pcd_world.shape[0])).astype(np.bool_)
        for instance in current_instance_info.keys():
            obj_to_world = current_instance_info[instance]['obj_to_world']
            length, width, height = current_instance_info[instance]['box_size']
            
            # padding the box
            if current_instance_info[instance]['class_name'] in PANDA_RIGID_DYNAMIC_CLASSES:
                length = length * 1.5
                width = width * 1.5
            
            pcd_world_homo = np.concatenate([pcd_world, np.ones_like(pcd_world[..., :1])], axis=-1)
            pcd_instance = (pcd_world_homo @ np.linalg.inv(obj_to_world).T)[..., :3]
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(pcd_instance, corners3d)
            pcd_instance_mask = np.logical_or(pcd_instance_mask, inbbox_mask)

            if inbbox_mask.sum() > 0:
                save_path = current_instance_info[instance]['save_path']
                storePly(save_path, pcd_instance[inbbox_mask], pcd_color[inbbox_mask], pcd_mask[inbbox_mask][:, None])

        pcd_bkgd_xyz = pcd_ego[~pcd_instance_mask]#pcd_world#[~pcd_instance_mask]
        pcd_bkgd_color = pcd_color[~pcd_instance_mask]
        pcd_bkgd_mask = pcd_mask[~pcd_instance_mask]
        ply_path = f"{lidar_background_dir}/{str(frame_idx).zfill(6)}.ply"
        storePly(ply_path, pcd_bkgd_xyz, pcd_bkgd_color, pcd_bkgd_mask[:, None])

    def save_poses(self, scene_data: Sequence, scene_idx, num_frames):
        ego_poses = []
        for frame_idx in tqdm(range(num_frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True):
            lidar_poses = scene_data.lidar.poses[frame_idx]
            ego_pose = geometry._heading_position_to_mat(lidar_poses['heading'], lidar_poses['position'])

            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/"
                + f"{str(frame_idx).zfill(6)}.txt",
                ego_pose,
            )
            ego_poses.append(ego_pose)
        return ego_poses

    def save_cam_poses(self, scene_data: Sequence, scene_idx, num_frames, extrinsics):
        for frame_idx in tqdm(range(num_frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True):
            lidar_poses = scene_data.lidar.poses[frame_idx]
            ego_pose = geometry._heading_position_to_mat(lidar_poses['heading'], lidar_poses['position'])
            for cam_id, cam_name in enumerate(self.cam_list):
                # ego_cam_pose = ego_pose @ extrinsics[cam_id]
                camera = scene_data.camera[cam_name]
                poses = camera.poses[frame_idx]
                ego_cam_pose = geometry._heading_position_to_mat(poses['heading'], poses['position'])
                np.savetxt(f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/{frame_idx:06d}_{cam_id}.txt", ego_cam_pose)

    def save_dynamic_mask(self, scene_data: Sequence, scene_idx, frame_idx, ego_poses, extrinsics, intrinsics, class_valid='all'):
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_mask"
        for cam_id, cam in enumerate(self.cam_list):
            dynamic_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cuboids = scene_data.cuboids[frame_idx]

            recorded_id = []
            for _, row in cuboids.iterrows():
                if row["label"] not in PANDA_DYNAMIC_CLASSES or row["stationary"]:
                    continue
                if not row["cuboids.sensor_id"] == -1:
                    if row["cuboids.sibling_id"] in recorded_id:
                        continue
                tx, ty, tz = row["position.x"], row["position.y"], row["position.z"]

                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(row["yaw"])
                s = np.math.sin(row["yaw"])

                obj_to_world = np.array([
                    [c, -s, 0, tx],
                    [s, c, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]])

                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                length, width, height = row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]
                pose_vehicle = np.linalg.inv(ego_poses[frame_idx]) @ obj_to_world
                corners_ego = compute_3d_box_corners([length, width, height], pose_vehicle)
                corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
                ego2cam = np.linalg.inv(extrinsics[cam_id])
                pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), intrinsics[cam_id], np.array([]))
                uv = uv.squeeze().astype(int)
                valid = (pts_cam[:, 2] > 0)
                edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]
                for i, j in edges:
                    if valid[i] and valid[j]:
                        try:
                            cv2.line(dynamic_mask, tuple(uv[i]), tuple(uv[j]), (0, 255, 0), 2)
                        except Exception as e:
                            print(f'==== error, type of ui[{i}]: {type(uv[i])}, type of ui[{j}]: {type(uv[j])}')
                            pass
                if valid.sum() >= 3:
                    hull = cv2.convexHull(uv[valid])
                    cv2.fillPoly(dynamic_mask, [hull], 255)
            dynamic_mask_path = os.path.join(mask_dir, f"{str(frame_idx).zfill(6)}_{str(cam_id)}.jpg")
            cv2.imwrite(dynamic_mask_path, dynamic_mask)

    def save_objects(self, scene_data: Sequence, scene_idx, num_frames, ego_poses, extrinsics, intrinsics):
        """Parse and save the objects annotation data.
        
        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            num_frames (int): Number of frames.
        """
        instances_info = {}
        track_info = dict()  # 以每个frame的一个bbox为单位 frame_id, track_id 记录LiDAR-synced和Camera_synced bboxes
        track_camera_visible = dict()  # 以每个camera的一个bbox为单位 frame_id, camera_id, track_id 记录这个camera看到了哪些物体
        trajectory_info = dict()  # 以每个track物体的一个bbox为单位 track_id, frame_id 记录LiDAR-synced boxes

        for frame_idx in range(num_frames):
            cuboids = scene_data.cuboids[frame_idx]
            track_info_cur_frame = dict()
            track_camera_visible_cur_frame = {cam_id: [] for cam_id, _ in enumerate(self.cam_list)}
            for _, row in cuboids.iterrows():
                if row["stationary"]:
                    continue
                str_id = row["uuid"]
                label = row["label"]
                if label not in PANDA_DYNAMIC_CLASSES:
                    continue

                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=str_id,
                        class_name=row["label"],
                        sibling_id=row["cuboids.sibling_id"],
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                            "stationary": [],
                        }
                    )

                # Box coordinates in vehicle frame.
                tx, ty, tz = row["position.x"], row["position.y"], row["position.z"]

                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(row["yaw"])
                s = np.math.sin(row["yaw"])

                obj_to_world = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                l, w, h = row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]
                instances_info[str_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[str_id]['frame_annotations']['obj_to_world'].append(obj_to_world.tolist())
                instances_info[str_id]['frame_annotations']['box_size'].append([l, w, h])
                instances_info[str_id]['frame_annotations']['stationary'].append(row["stationary"])

                obj_id = str_id
                if obj_id not in trajectory_info.keys():
                    trajectory_info[obj_id] = dict()
                track_info_cur_frame[obj_id] = dict()
                obj_to_ego = np.linalg.inv(ego_poses[frame_idx]) @ obj_to_world
                lidar_synced_box = dict()
                lidar_synced_box['height'] = h
                lidar_synced_box['width'] = w
                lidar_synced_box['length'] = l
                lidar_synced_box['center_x'] = obj_to_ego[0, 3]
                lidar_synced_box['center_y'] = obj_to_ego[1, 3]
                lidar_synced_box['center_z'] = obj_to_ego[2, 3]
                lidar_synced_box['rotation_matrix'] = obj_to_ego[:3, :3]
                quat = Rotation.from_matrix(obj_to_ego[:3, :3]).as_quat()
                lidar_synced_box['heading'] = quaternion_yaw(quat)
                lidar_synced_box['label'] = to_notr_label(row["label"])
                # lidar_synced_box['speed'] = np.linalg.norm(obj.speed)
                lidar_synced_box['timestamp'] = scene_data.camera["front_camera"].timestamps[frame_idx]
                track_info_cur_frame[obj_id]['lidar_box'] = lidar_synced_box
                trajectory_info[obj_id][f'{frame_idx:06d}'] = lidar_synced_box

                for cam_id, _ in enumerate(self.cam_list):
                    corners_ego = compute_3d_box_corners([l, w, h], obj_to_ego)
                    corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
                    ego2cam = np.linalg.inv(extrinsics[cam_id])
                    pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                    uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), intrinsics[cam_id],
                                              np.array([]))
                    uv = uv.squeeze()
                    inside = (uv[:, 0] >= 0) & (uv[:, 0] < img_width) & (uv[:, 1] >= 0) & (uv[:, 1] < img_height) & (
                            pts_cam[:, 2] > 0)
                    if inside.any():
                        track_camera_visible_cur_frame[cam_id].append(obj_id)

            track_info[f'{frame_idx:06d}'] = track_info_cur_frame
            track_camera_visible[f'{frame_idx:06d}'] = track_camera_visible_cur_frame

        self.reset_information_for_trajectory(trajectory_info, track_camera_visible, track_info, ego_poses)
        track_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/track/"
        with open(os.path.join(track_dir, "track_info.pkl"), 'wb') as f:
            pickle.dump(track_info, f)

        with open(os.path.join(track_dir, "track_camera_visible.pkl"), 'wb') as f:
            pickle.dump(track_camera_visible, f)

        with open(os.path.join(track_dir, "trajectory.pkl"), 'wb') as f:
            pickle.dump(trajectory_info, f)

        with open(os.path.join(track_dir, "track_ids.json"), 'w') as f:
            json.dump({t: i for i, t in enumerate(trajectory_info.keys())}, f, indent=2)

        return instances_info

    def reset_information_for_trajectory(self, trajectory_info, track_camera_visible, track_info, ego_frame_poses):
        # reset information for trajectory
        # poses, stationary, symmetric, deformable
        for obj_id in tqdm(trajectory_info.keys(), desc="Resetting trajectory information"):
            # for obj_id in trajectory_info.keys():
            new_trajectory = dict()
            trajectory = trajectory_info[obj_id]
            trajectory = dict(sorted(trajectory.items(), key=lambda item: item[0]))

            dims = []
            frames = []
            timestamps = []
            poses_vehicle = []
            poses_world = []
            speeds = []

            for frame_id, bbox in trajectory.items():
                label = bbox['label']
                dims.append([bbox['height'], bbox['width'], bbox['length']])
                frames.append(int(frame_id))
                timestamps.append(bbox['timestamp'])
                # speeds.append(bbox['speed'])
                pose_vehicle = np.eye(4)
                pose_vehicle[:3, :3] = bbox['rotation_matrix']
                pose_vehicle[:3, 3] = np.array([bbox['center_x'], bbox['center_y'], bbox['center_z']])

                ego_pose = ego_frame_poses[int(frame_id)]
                pose_world = np.matmul(ego_pose, pose_vehicle)

                poses_vehicle.append(pose_vehicle.astype(np.float32))
                poses_world.append(pose_world.astype(np.float32))

            dims = np.array(dims).astype(np.float32)
            dim = np.max(dims, axis=0)
            poses_vehicle = np.array(poses_vehicle).astype(np.float32)
            poses_world = np.array(poses_world).astype(np.float32)
            actor_world_postions = poses_world[:, :3, 3]
            distance = np.linalg.norm(actor_world_postions[0] - actor_world_postions[-1])
            # dynamic = (np.any(np.std(actor_world_postions, axis=0) > 0.5) or distance > 2) and np.mean(speeds) > 0.001

            new_trajectory['label'] = label
            new_trajectory['height'], new_trajectory['width'], new_trajectory['length'] = dim[0], dim[1], dim[2]
            new_trajectory['poses_vehicle'] = poses_vehicle
            new_trajectory['timestamps'] = timestamps
            new_trajectory['frames'] = frames
            # new_trajectory['speeds'] = speeds
            new_trajectory['symmetric'] = (label != 'pedestrian')
            new_trajectory['deformable'] = (label == 'pedestrian')
            new_trajectory['stationary'] = False#not dynamic
            trajectory_info[obj_id] = new_trajectory

        to_be_remove = []
        for obj_id, traj in trajectory_info.items():
            if traj['stationary']:
                to_be_remove.append(obj_id)
        for obj_id in to_be_remove:
            if obj_id in trajectory_info:
                del trajectory_info[obj_id]

    def process_objects_for_lidar(self, instances_info, num_frames):
        frame_instances = {}
        # solve duplicated objects from different lidars
        duplicated_id_pairs = []
        for k, v in instances_info.items():
            if v["sibling_id"] != '-':
                # find if the pair is already in the list
                if (v["id"], v["sibling_id"]) in duplicated_id_pairs or (v["sibling_id"],
                                                                         v["id"]) in duplicated_id_pairs:
                    continue
                else:
                    duplicated_id_pairs.append((v["id"], v["sibling_id"]))

        for pair in duplicated_id_pairs:
            # check if all in the pair are in the instances_info
            if pair[0] not in instances_info:
                # print(f"WARN: {pair[0]} not in instances_info")
                continue
            elif pair[1] not in instances_info:
                # print(f"WARN: {pair[1]} not in instances_info")
                continue
            else:
                # keep the longer one in pairs
                if len(instances_info[pair[0]]['frame_annotations']['frame_idx']) > \
                        len(instances_info[pair[1]]['frame_annotations']['frame_idx']):
                    instances_info.pop(pair[1])
                else:
                    instances_info.pop(pair[0])

        # rough filter stationary objects
        # if all the annotations of an object are stationary, remove it
        static_ids = []
        for k, v in instances_info.items():
            if all(v['frame_annotations']['stationary']):
                static_ids.append(v['id'])
        print(f"INFO: {len(static_ids)} static objects removed")
        for static_id in static_ids:
            instances_info.pop(static_id)
        print(f"INFO: Final number of objects: {len(instances_info)}")

        # update frame_instances
        for frame_idx in range(num_frames):
            # must ceate a object for each frame
            frame_instances[frame_idx] = []
            for k, v in instances_info.items():
                if frame_idx in v['frame_annotations']['frame_idx']:
                    frame_instances[frame_idx].append(v["id"])

        return instances_info, frame_instances

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/track", exist_ok=True)
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/sky_mask", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/intrinsics", exist_ok=True)
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
            if "lidar_forward" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar_forward", exist_ok=True)
            if "3dbox_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/3dbox_vis", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_mask", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
            if "objects_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances/debug_vis", exist_ok=True)
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="split of the dataset, e.g. training, validation, testing, please specify the split name for different dataset",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    # priority: scene_ids > split_file > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--max_frame_limit",
        type=int,
        default=300,
        help="maximum number of frames to be processed in a dataset, in nuplan dataset, \
            the scene duration super long, we can limit the number of frames to be processed, \
                this argument is used only for nuplan dataset",
    )
    parser.add_argument(
        "--start_frame_idx",
        type=int,
        default=1000,
        help="We skip the first start_frame_idx frames to avoid ego static frames",
    )
    parser.add_argument(
        "--interpolate_N",
        type=int,
        default=0,
        help="Interpolate to get frames at higher frequency, this is only used for nuscene dataset",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
    )
    args = parser.parse_args()
    
    scene_ids_list = args.scene_ids
    
    scene_ids_list = [str(scene_id).zfill(3) for scene_id in scene_ids_list]
    dataset_processor = PandaSetProcessor(
        load_dir=args.data_root,
        save_dir=args.target_dir,
        process_keys=args.process_keys,
        process_id_list=scene_ids_list,
        workers=args.workers,
    )

    for scene_id in scene_ids_list:
        dataset_processor.convert_one(scene_id)

    # dataset_processor.convert()

