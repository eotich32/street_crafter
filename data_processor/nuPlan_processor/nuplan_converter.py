from tqdm import tqdm
import json
import os
import sys
from typing import Dict, List
import argparse
import pickle

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

nuplan_dir = os.path.abspath('nuplan-devkit')
if nuplan_dir not in sys.path:
    sys.path.append(nuplan_dir)

from nuplan.database.nuplan_db.nuplan_scenario_queries import get_images_from_lidar_tokens
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

# from datasets.tools.multiprocess_utils import track_parallel_progress
from geometry import get_corners, project_camera_points_to_image
# from utils.visualization import color_mapper, dump_3d_bbox_on_image
from nuplan_utils import get_egopose3d_for_lidarpc_token_from_db, get_tracked_objects_for_lidarpc_token_from_db
import cv2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from waymo_processor.waymo_helpers import load_track, load_calibration, load_ego_poses, get_object, \
    project_label_to_image, project_label_to_mask, draw_3d_box_on_img, opencv2camera
from types import SimpleNamespace

NUPLAN_LABELS = [
    'vehicle', 'pedestrian', 'bicycle'
]
NUPLAN_NONRIGID_DYNAMIC_CLASSES = [
    'pedestrian', 'bicycle'
]
NUPLAN_RIGID_DYNAMIC_CLASSES = [
    'vehicle'
]
NUPLAN_DYNAMIC_CLASSES = NUPLAN_NONRIGID_DYNAMIC_CLASSES + NUPLAN_RIGID_DYNAMIC_CLASSES
img_width, img_height = 1920, 1080  # 去畸变后也是这个，没改变


def undistort_image(img_path, K, D, dst_size=(img_width, img_height)):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((dst_size[1], dst_size[0], 3), np.uint8), K
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0, newImgSize=dst_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dst_size, cv2.CV_16SC2)
    dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return dst, new_K


def quaternion_yaw(q) -> float:
    """从四元数计算偏航角(绕Z轴的旋转角度)"""
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return yaw


def dict_to_namespace(data):
    """递归将字典转换为 SimpleNamespace 对象"""
    if isinstance(data, dict):
        # 递归处理每个键值对
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in data.items()
        })
    elif isinstance(data, list):
        # 处理列表中的元素（若包含字典）
        return [dict_to_namespace(item) for item in data]
    else:
        # 普通值直接返回
        return data


def compute_3d_box_corners(dim, pose):
    l, w, h = dim
    x = np.array([-l, l, l, -l, -l, l, l, -l]) / 2
    y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    z = np.array([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    pts = np.stack([x, y, z, np.ones(8)], axis=0)
    pts = (pose @ pts)[:3, :].T
    return pts


class NuPlanProcessor(object):
    """Process NUPLAN Dataset

    NuPlan Datasets provides 8 cameras and Merged Lidar data.
    Cameras works in 10Hz and Lidar works in 20Hz. Thus we process at 10Hz.
    The duration of each scene is around 8 mins, resulting in ~5000 frames.
    We only process the first max_frame_limit(default=300) frames.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data in processed format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
        process_keys (list, optional): List of keys to process. Default: ["images", "lidar", "calib", "dynamic_masks", "objects"]
        process_log_list (list, optional): List of scene indices to process. Default: None
    """

    def __init__(
            self,
            load_dir='data/nuplan/raw',
            save_dir='data/nuplan/processed/test',
            prefix='mini',
            start_frame_idx=1000,
            max_frame_limit=300,
            process_keys=[
                "images",
                "pose",
                "calib",
                "lidar",
                "dynamic_masks",
                "objects"
            ],
            process_id_list=None,
            workers=64,
    ):
        self.HW = (img_height, img_width)
        print("Raw Image Resolution: ", self.HW)
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)
        self.start_frame_idx = start_frame_idx
        print("We will skip the first {} frames".format(self.start_frame_idx))
        self.max_frame_limit = max_frame_limit
        print("We will process the first {} frames each scene".format(self.max_frame_limit))
        # the lidar data is collected at 20Hz, we need to downsample to 10Hz to match the camera data
        self.lidar_idxs = range(self.start_frame_idx, self.start_frame_idx + self.max_frame_limit * 2, 2)

        # NUPLAN Provides 8 cameras
        self.cam_list = [  # {frame_idx}_{cam_id}.jpg
            "CAM_F0",  # "xxx_0.jpg"
            "CAM_L0",  # "xxx_1.jpg"
            "CAM_R0",  # "xxx_2.jpg"
            "CAM_L1",  # "xxx_3.jpg"
            "CAM_R1",  # "xxx_4.jpg"
            "CAM_L2",  # "xxx_5.jpg"
            "CAM_R2",  # "xxx_6.jpg"
            "CAM_B0"  # "xxx_7.jpg"
        ]

        self.sensor_blobs_dir = os.path.join(load_dir, 'nuplan-v1.1', 'sensor_blobs')
        self.split_dir = os.path.join(load_dir, 'nuplan-v1.1', 'splits', prefix)
        self.nuplandb_wrapper = NuPlanDBWrapper(
            data_root=os.path.join(load_dir, 'nuplan-v1.1'),
            map_root=os.path.join(load_dir, 'maps'),
            db_files=self.split_dir,
            map_version='nuplan-maps-v1.0',
        )

        process_log_list = []
        for idx in process_id_list:
            process_log_list.append(self.nuplandb_wrapper.log_names[idx])
        self.process_log_list = process_log_list

        self.save_dir = save_dir
        self.workers = int(workers)
        self.create_folder()

    # def convert(self):
    #     """Convert action."""
    #     print("Start converting ...")
    #     if self.process_log_list is None:
    #         id_list = range(len(self))
    #     else:
    #         id_list = self.process_log_list
    #     track_parallel_progress(self.convert_one, id_list, self.workers)
    #     print("\nFinished ...")

    def convert_one(self, scene_log_name):
        """Convert action for single file."""
        # get log db
        log_db = self.nuplandb_wrapper.get_log_db(scene_log_name)

        # since lidar and images are captured at different frequency
        # we find the best start frame that lidar and images matches the best
        # lidar_idx:[0]   1   [2]   3   [4]   5   [6]
        # timestamp: 0   0.05 0.1  0.15 0.2  0.25 0.3
        # lidar_pc:  |    |    |    |    |    |    |
        # Images:    |         |         |         |
        # NOTE: the best match should be the frame with the closest timestamp to the lidar_pc (e.g. [0] [2] [4] [6])
        # calulate time shift of original start frame
        lidar_pc = log_db.lidar_pc[self.start_frame_idx]
        images = get_images_from_lidar_tokens(
            log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
            tokens=[lidar_pc.token],
            channels=self.cam_list,
        )
        images_timestamps = [image.timestamp for image in images]
        lidar_timestamp = lidar_pc.timestamp
        no_shift_time_diff = [abs(lidar_timestamp - timestamp) for timestamp in images_timestamps]
        # calulate time shift of original start frame + 1
        lidar_pc = log_db.lidar_pc[self.start_frame_idx + 1]
        images = get_images_from_lidar_tokens(
            log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
            tokens=[lidar_pc.token],
            channels=self.cam_list,
        )
        images_timestamps = [image.timestamp for image in images]
        lidar_timestamp = lidar_pc.timestamp
        shift_time_diff = [abs(lidar_timestamp - timestamp) for timestamp in images_timestamps]

        if sum(no_shift_time_diff) > sum(shift_time_diff):
            self.lidar_idxs = [idx + 1 for idx in self.lidar_idxs]
        # else:
        #     lidar_idxs = self.lidar_idxs
        with open(os.path.join(f"{self.save_dir}", "lidar", "raw_lidar_idxs.pkl"), 'wb') as f:
            pickle.dump(self.lidar_idxs, f)

        new_intrinsics, img_timestamps = self.save_image(log_db, self.lidar_idxs)
        self.save_timestamps(log_db, self.lidar_idxs, img_timestamps)
        extrinsics = self.save_calib(log_db, new_intrinsics)
        ego_poses = self.save_pose(log_db, self.lidar_idxs, extrinsics)
        track_info, track_camera_visible, trajectory = self.save_objects(log_db, self.lidar_idxs, ego_poses,
                                                                         new_intrinsics, extrinsics)
        self.save_dynamic_mask(track_info, track_camera_visible, trajectory, new_intrinsics, extrinsics)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_log_list)

    def save_image(self, log_db: NuPlanDB, lidar_idxs: List[int]):
        """Parse and save the images in jpg format."""
        img_timestamps = {}  # # 第一级key为帧号，第二级key为相机id，value为该帧该相机对应的时间戳
        cameras = log_db.camera
        intrinsics, distortions = {}, {}
        new_intrinsics = {}
        for cam in cameras:
            channel = cam.channel
            intrinsics[channel] = np.array(cam.intrinsic)
            distortions[channel] = np.array(cam.distortion)
        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in tqdm(enumerate(lidar_idxs), total=len(lidar_idxs), desc="Processing images"):
            if frame_idx >= self.max_frame_limit:
                break
            img_timestamps[frame_idx] = {}
            lidar_pc = lidar_pcs[lidar_idx]

            images = get_images_from_lidar_tokens(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                tokens=[lidar_pc.token],
                channels=self.cam_list,
            )

            image_cnt = 0

            for cam_id, image in enumerate(images):
                if cam_id not in img_timestamps[frame_idx]:
                    img_timestamps[frame_idx][cam_id] = {}
                img_timestamps[frame_idx][cam_id] = image.timestamp
                raw_image_path = os.path.join(self.sensor_blobs_dir, image.filename_jpg)
                image_save_path = f"{self.save_dir}/images/{str(frame_idx).zfill(6)}_{cam_id}.jpg"

                channel = self.cam_list[cam_id]
                undistorted, new_K = undistort_image(raw_image_path, intrinsics[channel], distortions[channel],
                                                     dst_size=(img_width, img_height))
                cv2.imwrite(image_save_path, undistorted)
                # os.system(f'cp {raw_image_path} {image_save_path.replace("images","raw_images")}')
                new_intrinsics[cam_id] = new_K
                image_cnt += 1

            assert image_cnt == len(self.cam_list), \
                f"Image number, camera number mismatch: {image_cnt} != {len(self.cam_list)}"
        return new_intrinsics, img_timestamps

    def save_timestamps(self, log_db, lidar_idxs, img_timestamps):
        lidar_pcs = log_db.lidar_pc
        timestamps = {'FRAME': {}} | {cam: {} for cam in self.cam_list}
        for frame_idx, lidar_idx in tqdm(enumerate(lidar_idxs), total=len(lidar_idxs), desc="Processing timestamps"):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            timestamps['FRAME'][str(frame_idx).zfill(6)] = lidar_pc.timestamp / 1e6
            for cam_id, _ in enumerate(self.cam_list):
                timestamps[self.cam_list[cam_id]][str(frame_idx).zfill(6)] = img_timestamps[frame_idx][cam_id] / 1e6
        with open(f"{self.save_dir}/timestamps.json", 'w') as f:
            json.dump(timestamps, f, indent=2)

    def get_cameras_calib(self, log_db: NuPlanDB):
        """Get the camera calibration."""
        cameras = log_db.camera
        extrinsics, intrinsics, distortions = {}, {}, {}
        for cam in cameras:
            channel = cam.channel
            extrinsic = Quaternion(cam.rotation).transformation_matrix
            extrinsic[:3, 3] = np.array(cam.translation)
            extrinsics[channel] = extrinsic
            intrinsic = np.array(cam.intrinsic)
            intrinsics[channel] = intrinsic
            distortions[channel] = np.array(cam.distortion)

        return extrinsics, intrinsics, distortions

    def save_calib(self, log_db: NuPlanDB, new_intrinsics):
        """Parse and save the calibration data."""
        extrinsics, intrinsics, distortions = self.get_cameras_calib(log_db)
        extrinsics_by_cam_id = {}
        for channel in tqdm(self.cam_list, desc="Processing cameras"):
            cam_id = self.cam_list.index(channel)

            intrinsic = intrinsics[channel]
            fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            k1, k2, p1, p2, k3 = distortions[channel]

            extrinsic = extrinsics[channel]
            np.savetxt(
                f"{self.save_dir}/extrinsics/"
                + f"{str(cam_id)}.txt",
                extrinsic
            )
            extrinsics_by_cam_id[cam_id] = extrinsic
            if new_intrinsics is None:
                Ks = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3])
            else:
                intr = new_intrinsics[cam_id]
                Ks = np.array([intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]])
            np.savetxt(
                f"{self.save_dir}/intrinsics/"
                + f"{str(cam_id)}.txt",
                Ks
            )
        return extrinsics_by_cam_id

    def save_pose(self, log_db: NuPlanDB, lidar_idxs: List[int], extrinsics):
        """Parse and save the pose data."""
        lidar_pcs = log_db.lidar_pc
        ego_poses = []
        for frame_idx, lidar_idx in tqdm(enumerate(lidar_idxs), total=len(lidar_idxs), desc="Processing ego poses"):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]

            ego_pose = get_egopose3d_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )

            np.savetxt(
                f"{self.save_dir}/ego_pose/"
                + f"{str(frame_idx).zfill(6)}.txt",
                ego_pose
            )
            for cam_id, _ in enumerate(self.cam_list):
                ego_cam_pose = ego_pose @ extrinsics[cam_id]
                np.savetxt(f"{self.save_dir}/ego_pose/{frame_idx:06d}_{cam_id}.txt", ego_cam_pose)
            ego_poses.append(ego_pose)
        return ego_poses

    def save_dynamic_mask(self, track_info, track_camera_visible, trajectory, new_intrinsics, extrinsics):
        for idx, (frame, track_info_frame) in tqdm(enumerate(track_info.items()), total=len(track_info),
                                                   desc="Processing dynamic masks"):
            track_camera_visible_cur_frame = track_camera_visible[frame]
            for cam_id, track_ids in track_camera_visible_cur_frame.items():
                dynamic_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                for track_id in track_ids:
                    object_tracklet = trajectory[track_id]
                    if object_tracklet['stationary']:
                        continue
                    pose_idx = object_tracklet['frames'].index(int(frame))
                    pose_vehicle = object_tracklet['poses_vehicle'][pose_idx]
                    height, width, length = object_tracklet['height'], object_tracklet['width'], object_tracklet[
                        'length']

                    corners_ego = compute_3d_box_corners([length, width, height], pose_vehicle)
                    corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
                    ego2cam = np.linalg.inv(extrinsics[cam_id])
                    pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                    uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), new_intrinsics[cam_id],
                                              np.array([]))
                    uv = uv.squeeze().astype(int)
                    valid = (pts_cam[:, 2] > 0)
                    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                             (4, 5), (5, 6), (6, 7), (7, 4),
                             (0, 4), (1, 5), (2, 6), (3, 7)]
                    for i, j in edges:
                        if valid[i] and valid[j]:
                            cv2.line(dynamic_mask, tuple(uv[i]), tuple(uv[j]), (0, 255, 0), 2)
                    if valid.sum() >= 3:
                        hull = cv2.convexHull(uv[valid])
                        cv2.fillPoly(dynamic_mask, [hull], 255)
                dynamic_mask_path = os.path.join(self.save_dir, "dynamic_mask", f"{str(frame).zfill(6)}_{cam_id}.jpg")
                cv2.imwrite(dynamic_mask_path, dynamic_mask)

    def save_objects(self, log_db: NuPlanDB, lidar_idxs: List[int], ego_poses, new_intrinsics, extrinsics):
        """Parse and save the objects annotation data."""
        track_info = dict()  # 以每个frame的一个bbox为单位 frame_id, track_id 记录LiDAR-synced和Camera_synced bboxes
        track_camera_visible = dict()  # 以每个camera的一个bbox为单位 frame_id, camera_id, track_id 记录这个camera看到了哪些物体
        trajectory_info = dict()  # 以每个track物体的一个bbox为单位 track_id, frame_id 记录LiDAR-synced boxes

        lidar_pcs = log_db.lidar_pc
        for frame_idx, lidar_idx in tqdm(enumerate(lidar_idxs), total=len(lidar_idxs),
                                         desc="Processing dynamic objects"):
            if frame_idx >= self.max_frame_limit:
                break
            lidar_pc = lidar_pcs[lidar_idx]
            track_info_cur_frame = dict()
            track_camera_visible_cur_frame = {cam_id: [] for cam_id, _ in enumerate(self.cam_list)}

            objects_generator = get_tracked_objects_for_lidarpc_token_from_db(
                log_file=os.path.join(self.split_dir, log_db.log_name + '.db'),
                token=lidar_pc.token
            )
            objects = [obj for obj in objects_generator if obj.category in NUPLAN_DYNAMIC_CLASSES]

            for obj in objects:
                obj_id = obj.track_token
                if obj_id not in trajectory_info.keys():
                    trajectory_info[obj_id] = dict()
                track_info_cur_frame[obj_id] = dict()
                obj_to_world = obj.pose
                l, w, h = obj.box_size

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
                lidar_synced_box['label'] = obj.category
                lidar_synced_box['speed'] = np.linalg.norm(obj.speed)
                lidar_synced_box['timestamp'] = obj.timestamp
                track_info_cur_frame[obj_id]['lidar_box'] = lidar_synced_box
                trajectory_info[obj_id][f'{frame_idx:06d}'] = lidar_synced_box

                for cam_id, _ in enumerate(self.cam_list):
                    corners_ego = compute_3d_box_corners([l, w, h], obj_to_ego)
                    corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
                    ego2cam = np.linalg.inv(extrinsics[cam_id])
                    pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                    uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), new_intrinsics[cam_id],
                                              np.array([]))
                    uv = uv.squeeze()
                    inside = (uv[:, 0] >= 0) & (uv[:, 0] < img_width) & (uv[:, 1] >= 0) & (uv[:, 1] < img_height) & (
                                pts_cam[:, 2] > 0)
                    if inside.any():
                        track_camera_visible_cur_frame[cam_id].append(obj_id)

            track_info[f'{frame_idx:06d}'] = track_info_cur_frame
            track_camera_visible[f'{frame_idx:06d}'] = track_camera_visible_cur_frame

        self.reset_information_for_trajectory(trajectory_info, track_camera_visible, track_info, ego_poses)
        track_dir = f"{self.save_dir}/track"
        with open(os.path.join(track_dir, "track_info.pkl"), 'wb') as f:
            pickle.dump(track_info, f)

        with open(os.path.join(track_dir, "track_camera_visible.pkl"), 'wb') as f:
            pickle.dump(track_camera_visible, f)

        with open(os.path.join(track_dir, "trajectory.pkl"), 'wb') as f:
            pickle.dump(trajectory_info, f)

        with open(os.path.join(track_dir, "track_ids.json"), 'w') as f:
            json.dump({t: i for i, t in enumerate(trajectory_info.keys())}, f, indent=2)
        return track_info, track_camera_visible, trajectory_info

    def create_folder(self):
        if "images" in self.process_keys:
            os.makedirs(f"{self.save_dir}/images", exist_ok=True)
            os.makedirs(f"{self.save_dir}/sky_masks", exist_ok=True)
        if "pose" in self.process_keys:
            os.makedirs(f"{self.save_dir}/ego_pose", exist_ok=True)
        if "calib" in self.process_keys:
            os.makedirs(f"{self.save_dir}/extrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/intrinsics", exist_ok=True)
        if "lidar" in self.process_keys:
            os.makedirs(f"{self.save_dir}/lidar", exist_ok=True)
        if "dynamic_masks" in self.process_keys:
            os.makedirs(f"{self.save_dir}/dynamic_mask", exist_ok=True)
        if "objects" in self.process_keys:
            os.makedirs(f"{self.save_dir}/track", exist_ok=True)

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
                speeds.append(bbox['speed'])
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
            dynamic = (np.any(np.std(actor_world_postions, axis=0) > 0.5) or distance > 2) and np.mean(speeds) > 0.001

            new_trajectory['label'] = label
            new_trajectory['height'], new_trajectory['width'], new_trajectory['length'] = dim[0], dim[1], dim[2]
            new_trajectory['poses_vehicle'] = poses_vehicle
            new_trajectory['timestamps'] = timestamps
            new_trajectory['frames'] = frames
            new_trajectory['speeds'] = speeds
            new_trajectory['symmetric'] = (label != 'pedestrian')
            new_trajectory['deformable'] = (label == 'pedestrian')
            new_trajectory['stationary'] = not dynamic
            trajectory_info[obj_id] = new_trajectory

            if not dynamic: # 上面的过滤会有速度为0的被视为动态对象，这里强行删除一下。因为nuplan是自动标注，很多静态对象被标注出来，速度为0
                for frame_id in range(len(ego_frame_poses)):
                    if obj_id in track_info[str(frame_id).zfill(6)]:
                        del track_info[str(frame_id).zfill(6)][obj_id]
                    for cam_id, _ in enumerate(self.cam_list):
                        visible_objs = track_camera_visible[str(frame_id).zfill(6)][cam_id]
                        track_camera_visible[str(frame_id).zfill(6)][cam_id] = list(filter(lambda x: x != obj_id, visible_objs))

        to_be_remove = []
        for obj_id, traj in trajectory_info.items():
            if traj['stationary']:
                to_be_remove.append(obj_id)
        for obj_id in to_be_remove:
            if obj_id in trajectory_info:
                del trajectory_info[obj_id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuplan_root', required=True, help='nuPlan dataset root (data_root)')
    parser.add_argument('--log_name', required=True, help='log name token')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--start_frame', type=int, default=0, help='从哪一帧开始提取')
    parser.add_argument('--num_frames', type=int, default=200, help='提取的帧数')

    args = parser.parse_args()
    processor = NuPlanProcessor(load_dir=args.nuplan_root, save_dir=args.save_dir, start_frame_idx=args.start_frame,
                                max_frame_limit=args.num_frames, process_id_list=[0])
    processor.convert_one(args.log_name)
# python nuplan_preprocess.py --nuplan_root /mnt/data/dataset/nuPlan/raw --log_name 2021.05.12.22.00.38_veh-35_01008_01518 --save_dir /mnt/data/dataset/nuPlan/processed/01518_frame_1000_1200 --start_frame 1000 --num_frames 200
