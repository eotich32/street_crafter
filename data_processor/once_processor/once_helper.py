import os
import numpy as np
import pickle
import cv2
from typing import Dict, List, Tuple


# ---------- 工具函数 ----------
def load_track_once(track_dir: str):
    """加载 ONCE 轨迹信息"""
    with open(os.path.join(track_dir, 'track_info.pkl'), 'rb') as f:
        track_info = pickle.load(f)
    with open(os.path.join(track_dir, 'track_camera_visible.pkl'), 'rb') as f:
        track_camera_visible = pickle.load(f)
    with open(os.path.join(track_dir, 'trajectory.pkl'), 'rb') as f:
        trajectory = pickle.load(f)
    return track_info, track_camera_visible, trajectory


def load_ego_poses_once(ego_pose_dir: str):
    """加载自车位姿（已中心化）"""
    ego_files = sorted([f for f in os.listdir(ego_pose_dir) if not '_' in f])
    ego_poses = [np.loadtxt(os.path.join(ego_pose_dir, f)) for f in ego_files]
    return np.array(ego_poses)


def load_cam_poses_once(ego_pose_dir: str):
    """加载每帧每个相机的位姿"""
    cam_files = sorted([f for f in os.listdir(ego_pose_dir) if '_' in f])
    cam_poses = {}
    for f in cam_files:
        frame, cam = f.split('.')[0].split('_')
        pose = np.loadtxt(os.path.join(ego_pose_dir, f))
        cam_poses.setdefault(int(frame), {})[int(cam)] = pose
    return cam_poses


def load_calib_once(extrinsics_dir: str, intrinsics_dir: str):
    """加载相机内外参"""
    extrinsics = []
    intrinsics = []
    cam_ids = sorted([int(f.split('.')[0]) for f in os.listdir(extrinsics_dir)])
    for cam_id in cam_ids:
        extr = np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt"))
        intr = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"))
        extrinsics.append(extr)
        intrinsics.append(intr)
    return extrinsics, intrinsics


def project_lidar_to_image_once(xyz, T_velo2cam, K, img_size=(1920, 1020)):
    """将点云投影到图像平面"""
    pts_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
             (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int), mask, pts_cam[inside, 2]


def bbox_to_corners_3d_once(bbox: np.ndarray):
    """从 [x, y, z, l, w, h, yaw] 生成 8 个角点"""
    x, y, z, l, w, h, yaw = bbox
    cos, sin = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return (R @ corners.T).T + np.array([x, y, z])


def draw_3d_box_once(uv: np.ndarray, img: np.ndarray, color=(0, 255, 0), thickness=2):
    """绘制 3D 框"""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, thickness)