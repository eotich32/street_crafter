import os
import numpy as np
import pickle
import argparse
import json
from tqdm import tqdm
# from utils.box_utils import bbox_to_corner3d
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
import cv2
import sys
from pyquaternion import Quaternion

nuplan_dir = os.path.abspath('nuplan-devkit')
if nuplan_dir not in sys.path:
    sys.path.append(nuplan_dir)
from nuplan.database.nuplan_db_orm.utils import load_pointcloud_from_pc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)
# from waymo_processor.waymo_helpers import load_camera_info


# ---------------- 工具 ----------------
img_width, img_height = 1920, 1080  # 去畸变后也是这个，没改变

# 固定映射（与图像导出阶段保持一致）
cam_name2id = {
    "CAM_F0": 0, "CAM_L0": 1, "CAM_R0": 2, "CAM_L1": 3,
    "CAM_R1": 4, "CAM_L2": 5, "CAM_R2": 6, "CAM_B0": 7
}
cam_names = [
    "CAM_F0",  # "xxx_0.jpg"
    "CAM_L0",  # "xxx_1.jpg"
    "CAM_R0",  # "xxx_2.jpg"
    "CAM_L1",  # "xxx_3.jpg"
    "CAM_R1",  # "xxx_4.jpg"
    "CAM_L2",  # "xxx_5.jpg"
    "CAM_R2",  # "xxx_6.jpg"
    "CAM_B0"  # "xxx_7.jpg"
]


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def quat_to_mat(w, x, y, z):
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def get_point_colors(points_ego, img, K, RT, H, W):
    """
    将 ego 系点云投影到图像，提取 RGB 颜色
    points_ego: (N, 3)
    img: (H, W, 3) uint8
    K: (3, 3)
    RT: (4, 4) ego → cam
    return: (N, 3) float32 [0, 1]
    """
    xyz_cam = points_ego @ RT[:3, :3].T + RT[:3, 3]
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = xyz_cam @ K.T
    z = xyz_pixel[:, 2]
    uv = xyz_pixel[:, :2] / (z[:, None] + 1e-6)
    u, v = uv[:, 0], uv[:, 1]

    valid_u = np.logical_and(u >= 0, u < W)
    valid_v = np.logical_and(v >= 0, v < H)
    valid_pixel = np.logical_and.reduce([valid_depth, valid_u, valid_v])

    u_int = np.clip(u[valid_pixel].astype(np.int32), 0, W - 1)
    v_int = np.clip(v[valid_pixel].astype(np.int32), 0, H - 1)

    colors = np.zeros((points_ego.shape[0], 3), dtype=np.float32)
    colors[valid_pixel] = img[v_int, u_int] / 255.0

    return colors, valid_pixel


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def project_numpy(xyz, K, RT, H, W):
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)


def build_lidar2ego(lidar_rec):
    """构建 lidar -> ego 的 4x4 矩阵"""
    R = quat_to_mat(lidar_rec.rotation[0], *lidar_rec.rotation[1:])  # w,x,y,z
    t = np.array(lidar_rec.translation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def storePly(path, xyz, rgb, mask):
    # set rgb to 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)

    # set mask to bool data type
    mask = mask.astype(np.bool_)

    # Define the dtype for the structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('mask', '?')
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb, mask), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def inbbox_points(points, corner3d):
    min_xyz = corner3d[0]
    max_xyz = corner3d[-1]
    return np.logical_and(
        np.all(points >= min_xyz, axis=-1),
        np.all(points <= max_xyz, axis=-1)
    )


def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]

    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d


def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(len(cam_name2id)):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(len(cam_name2id)):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(len(cam_names))]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = int(ego_pose_path.split('.')[0][-1])
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(len(cam_names))]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


def generate_depth_from_lidar(points_ego, K, D, E, H, W):
    """
    使用与 nuplan_converter.py 中一致的投影方式：
    - E 是 camera → ego 的 4x4 外参
    - K 是去畸变后的内参
    - D 是畸变系数（通常为0，因为已经undistort）
    """
    ego2cam = np.linalg.inv(E)
    pts_cam = (ego2cam[:3, :3] @ points_ego.T + ego2cam[:3, 3:4]).T  # (N, 3)

    # 投影到图像
    uv, _ = cv2.projectPoints(
        pts_cam,  # (N, 3)
        np.zeros((3, 1)),  # rvec
        np.zeros((3, 1)),  # tvec
        K, D
    )
    uv = uv.squeeze()  # (N, 2)

    # 过滤
    u, v = uv[:, 0], uv[:, 1]
    z = pts_cam[:, 2]
    valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u_int = np.clip(u[valid].astype(int), 0, W - 1)
    v_int = np.clip(v[valid].astype(int), 0, H - 1)
    z_valid = z[valid]

    # 去重：同一像素保留最近深度
    depth_dict = {}
    for ui, vi, zi in zip(u_int, v_int, z_valid):
        key = (vi, ui)
        if key not in depth_dict or zi < depth_dict[key]:
            depth_dict[key] = zi

    depth = np.full((H, W), np.finfo(np.float32).max, dtype=np.float32)
    for (vi, ui), zi in depth_dict.items():
        depth[vi, ui] = zi

    depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
    valid_mask = depth != 0
    valid_values = depth[valid_mask]

    return valid_mask, valid_values


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)  float32
    return: (H, W, 3) uint8
    """
    x = np.nan_to_num(depth)
    if minmax is None:
        mi = np.min(x[x > 0])
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def undistort_camera_matrix(K, D, img_size=(img_width, img_height)):
    """只算 new_K，不处理图像"""
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, img_size, alpha=0, newImgSize=img_size)
    return new_K


# ---------------- 主函数 ----------------
def export_lidar_pcd(nuplan_root, log_name, save_dir, skip_existing=False):
    # 在第一步处理中（nuplan_converter.py），雷达已经降采样和相机时间帧对齐，这里取雷达帧在原始数据库中的帧号
    with open(os.path.join(f"{save_dir}", "lidar", "raw_lidar_idxs.pkl"), 'rb') as f:
        raw_lidar_idxs = pickle.load(f)

    db_path = os.path.join(nuplan_root, 'nuplan-v1.1', 'splits', 'mini')

    db = NuPlanDB(data_root=db_path, load_path=log_name)

    cam_newK_dict = {}
    for cam in db.camera:
        K = cam.intrinsic_np
        D = cam.distortion_np
        new_K = undistort_camera_matrix(K, D, img_size=(img_width, img_height))
        cam_newK_dict[cam.channel] = new_K

    # ---------- 2. 加载辅助数据 ----------
    track_dir = os.path.join(save_dir, 'track')
    with open(os.path.join(track_dir, 'track_info.pkl'), 'rb') as f:
        track_info = pickle.load(f)
    with open(os.path.join(track_dir, 'trajectory.pkl'), 'rb') as f:
        trajectory = pickle.load(f)

    # ---------- 3. 输出目录（与 Waymo 侧一致） ----------
    lidar_dir = os.path.join(save_dir, 'lidar')
    background_dir = os.path.join(lidar_dir, 'background')
    actor_dir = os.path.join(lidar_dir, 'actor')
    depth_dir = os.path.join(lidar_dir, 'depth')
    makedirs(depth_dir)

    makedirs(background_dir)
    for track_id in trajectory.keys():
        makedirs(os.path.join(actor_dir, track_id))

    # ---------- 4. 构造 lidar->ego 外参（同一 log 只有一套） ----------
    lidar2ego = build_lidar2ego(db.lidar[0])

    # ---------- 5. 按 CAM_F0 时间戳找最近 LiDAR 帧 ----------
    lidar_pcs = db.lidar_pc
    lidar_ts = np.array([pc.timestamp for pc in lidar_pcs], dtype=np.int64)

    def nearest_lidar_idx(ts_us):
        return int(np.abs(lidar_ts - ts_us).argmin())

    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(actor_dir, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []

    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(save_dir)
    # ---------- 6. 逐帧处理 ----------
    for frame_idx, lidar_idx in tqdm(enumerate(raw_lidar_idxs), total=len(raw_lidar_idxs), desc="Processing lidar points"):
        pc = lidar_pcs[lidar_idx]
        main_lidar = None
        for lidar in db.lidar:
            if lidar.token == pc.lidar_token:
                main_lidar = lidar
        lidar_pose_in_ego = np.eye(4)
        lidar_pose_in_ego[:3,:3] = Quaternion(main_lidar.rotation).transformation_matrix[:3,:3]
        lidar_pose_in_ego[:3, 3] = np.array(main_lidar.translation)

        # 6.1 加载点云（lidar 系）
        lidar_data: LidarPointCloud = pc.load(db, lidar_idx)
        xyzs = lidar_data.points[:3].T  # (N,3)
        points_in_lidar = np.hstack([xyzs, np.ones((xyzs.shape[0], 1))])
        rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.uint8)
        xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)

        # 6.2 lidar -> ego
        points_ego = (lidar2ego @ np.hstack([xyzs, np.ones((xyzs.shape[0], 1))]).T).T[:, :3]
        for cam in db.camera:
            cam_name = cam.channel
            if cam_name not in cam_name2id:
                continue
            cam_id = cam_name2id[cam_name]
            image_filename = os.path.join(save_dir, 'images', f'{frame_idx:06d}_{cam_id}.jpg')
            image = cv2.imread(image_filename)[..., [2, 1, 0]].astype(np.uint8)
            H, W = image.shape[:2]
            lidar_pose_in_camera = np.linalg.inv(extrinsics[cam_id]) @ lidar_pose_in_ego
            points_in_camera = np.dot(lidar_pose_in_camera, points_in_lidar.T).T
            instrinsic = np.concatenate([intrinsics[cam_id], np.array([[0,0,0]]).T], axis=1)
            points_pixel = np.dot(instrinsic, points_in_camera.T).T
            uvs = points_pixel / (points_pixel[:, 2].reshape(-1, 1))
            u, v, d = uvs[:, 0].astype(int), uvs[:, 1].astype(int), points_in_camera[:, 2]
            valid_idx = np.where((u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0))[0]
            u, v, depth = u[valid_idx], v[valid_idx], points_in_camera[:, 2][valid_idx]
            rgbs[valid_idx] = image[v, u]

        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f'{frame_idx:06d}']
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue

            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(frame_idx)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]

            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)

            actor_mask = np.logical_or(actor_mask, inbbox_mask)

            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = rgbs[inbbox_mask]
            # masks_inbbox = masks[inbbox_mask]

            pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
            pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
            # pointcloud_actor[track_id]['mask'].append(masks_inbbox)

            # masks_inbbox = masks_inbbox[..., None]
            ply_actor_path = os.path.join(actor_dir, track_id, f'{frame_idx:06d}.ply')
            try:
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox,
                         np.ones((xyzs_inbbox.shape[0], 1), dtype=bool))  # masks_inbbox
            except:
                pass  # No pcd

        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        # masks_background = masks[~actor_mask]
        # masks_background = masks_background[..., None]
        ply_background_path = os.path.join(background_dir, f'{frame_idx:06d}.ply')

        storePly(ply_background_path, xyzs_background, rgbs_background,
                 np.ones((xyzs_background.shape[0], 1), dtype=bool))  # masks_background

        # ---------- 7. 为所有相机生成深度图 + 保存映射 ----------
        depth_dir = os.path.join(save_dir, 'lidar', 'depth')
        makedirs(depth_dir)

        # 遍历所有相机
        for cam in db.camera:
            cam_name = cam.channel
            if cam_name not in cam_name2id:
                continue
            cam_id = cam_name2id[cam_name]

            H, W = img_height, img_width
            new_K = cam_newK_dict[cam.channel]
            # depth_mask, depth_value = generate_depth_from_lidar(points_ego, K, RT, H, W)
            depth_mask, depth_value = generate_depth_from_lidar(
                points_ego,
                K=new_K,  # 去畸变后的内参
                D=np.zeros(5),  # 已经 undistort，畸变为0
                E=extrinsics[cam_id], #cam.trans_matrix,  # camera → ego
                H=img_height,
                W=img_width
            )
            np.savez_compressed(os.path.join(depth_dir, f"{frame_idx:06d}_{cam_id}.npz"),
                                mask=depth_mask,
                                value=depth_value)

            img_path = os.path.join(save_dir, 'images', f'{frame_idx:06d}_{cam_id}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((H, W, 3), dtype=np.uint8)

            # 黑底深度可视化
            depth_map = np.zeros((H, W), dtype=np.float32)
            depth_map[depth_mask] = depth_value
            depth_color, _ = visualize_depth_numpy(depth_map, minmax=[0, 80])

            # 叠加
            overlay = img.copy()
            overlay[depth_mask] = depth_color[depth_mask]
            cv2.imwrite(os.path.join(depth_dir, f"{frame_idx:06d}_{cam_id}.jpg"), overlay)


# ---------------- entry ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuplan_root', required=True)
    parser.add_argument('--log_name', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    export_lidar_pcd(args.nuplan_root, args.log_name, args.save_dir, args.skip_existing)
