import os
import numpy as np
import pickle
import argparse
import json
from tqdm import tqdm
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
#from utils.box_utils import bbox_to_corner3d
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
import cv2
from nuplan.database.nuplan_db_orm.utils import load_pointcloud_from_pc

# ---------------- 工具 ----------------
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
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

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

def inbbox_points(points, R, center, dim):
    """
    判断 ego 系点云是否在 ego 系框内
    points: (N,3)  ego
    R:      (3,3)  ego 系旋转矩阵（z 轴对应 heading）
    center: (3,)   ego
    dim:    [L,W,H]
    """
    pts_local = (R.T @ (points - center).T).T
    hl, hw, hh = np.asarray(dim) / 2
    return (np.abs(pts_local[:, 0]) <= hl) & \
           (np.abs(pts_local[:, 1]) <= hw) & \
           (np.abs(pts_local[:, 2]) <= hh)

def generate_depth_from_lidar(points_ego, K, RT, H, W):
    """将 ego 系点云投影到图像平面，生成稀疏深度图（与 Waymo 格式一致）"""
    xyz_cam = points_ego @ RT[:3, :3].T + RT[:3, 3]
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = xyz_cam @ K.T
    z = xyz_pixel[:, 2]
    uv = xyz_pixel[:, :2] / (z[:, None] + 1e-6)
    u, v = uv[:, 0], uv[:, 1]

    valid_u = np.logical_and(u >= 0, u < W)
    valid_v = np.logical_and(v >= 0, v < H)
    valid_pixel = np.logical_and.reduce([valid_depth, valid_u, valid_v])

    u_valid = u[valid_pixel]
    v_valid = v[valid_pixel]
    z_valid = z[valid_pixel]

    # 转成整数像素坐标
    u_int = np.clip(u_valid.astype(np.int32), 0, W - 1)
    v_int = np.clip(v_valid.astype(np.int32), 0, H - 1)

    # 用字典去重：同一像素只保留最近深度
    depth_dict = {}
    for ui, vi, zi in zip(u_int, v_int, z_valid):
        key = (vi, ui)
        if key not in depth_dict or zi < depth_dict[key]:
            depth_dict[key] = zi

    # 构造深度图
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

# ---------------- 主函数 ----------------
def export_lidar_pcd(nuplan_root, log_name, save_dir, skip_existing=False):

    db_path = os.path.join(nuplan_root, 'nuplan-v1.1', 'splits', 'mini')

    db = NuPlanDB(data_root=db_path, load_path=log_name)

    # ---------- 1. 读取 CAM_F0 时间戳 ----------
    ts_json_path = os.path.join(save_dir, 'timestamps.json')
    if not os.path.isfile(ts_json_path):
        raise FileNotFoundError(f'找不到 {ts_json_path}，请先运行图像导出步骤！')
    with open(ts_json_path, 'r') as f:
        ts_dict = json.load(f)
    if 'CAM_F0' not in ts_dict:
        raise KeyError('timestamps.json 中缺少 CAM_F0')
    cam_f0_ts_list = np.array([int(v) for v in ts_dict['CAM_F0'].values()], dtype=np.int64)

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

    # ---------- 6. 逐帧处理 ----------
    for cam_idx, cam_ts in enumerate(tqdm(cam_f0_ts_list, desc='CAM_F0 matched LiDAR')):
        lidar_idx = nearest_lidar_idx(cam_ts)
        pc = lidar_pcs[lidar_idx]

        # 6.1 加载点云（lidar 系）
        pc_obj = load_pointcloud_from_pc(
            nuplandb=db,
            token=pc.token,
            nsweeps=1,
            max_distance=1e6,
            min_distance=0.0,
            use_intensity=False,
            use_ring=False,
            use_lidar_index=False,
        )
        points_lid = pc_obj.points[:3].T  # (N,3)

        # 6.2 lidar -> ego
        points_ego = (lidar2ego @ np.hstack([points_lid, np.ones((points_lid.shape[0], 1))]).T).T[:, :3]

        # 6.3 actor 分割（ego 系下做）
        actor_mask = np.zeros(points_ego.shape[0], dtype=bool)
        frame_key = f'{cam_idx:06d}'
        if frame_key in track_info:
            for track_id, box in track_info[frame_key].items():
                center = np.array([box['center_x'], box['center_y'], box['center_z']])
                dim    = [box['length'], box['width'], box['height']]
                heading= box['heading']
                R      = Rotation.from_euler('z', heading).as_matrix()

                # 判断点是否在框内
                in_box = inbbox_points(points_ego, R, center, dim)
                actor_mask |= in_box

                # 保存 actor 点云（局部系）
                actor_points_ego = points_ego[in_box]
                if actor_points_ego.shape[0] == 0:
                    continue
                if cam_idx not in trajectory[track_id]['frames']:
                    continue
                pose_idx = trajectory[track_id]['frames'].index(cam_idx)
                T_ego_obj = np.linalg.inv(trajectory[track_id]['poses_vehicle'][pose_idx])
                ones = np.ones((actor_points_ego.shape[0], 1))
                actor_points_local = (T_ego_obj @ np.hstack([actor_points_ego, ones]).T).T[:, :3]

                storePly(os.path.join(actor_dir, track_id, f"{cam_idx:06d}.ply"),
                         actor_points_local,
                         np.zeros_like(actor_points_local),
                         np.ones((actor_points_local.shape[0], 1), dtype=bool))

        # 6.4 保存背景点云（ego 系）
        background_points = points_ego[~actor_mask]
        storePly(os.path.join(background_dir, f"{cam_idx:06d}.ply"),
                 background_points,
                 np.zeros_like(background_points),
                 np.ones((background_points.shape[0], 1), dtype=bool))

        # ---------- 7. 为所有相机生成深度图 + 保存映射 ----------
        depth_dir = os.path.join(save_dir, 'lidar', 'depth'); makedirs(depth_dir)

        # 固定映射（与图像导出阶段保持一致）
        cam_name2id = {
            "CAM_L0": 0, "CAM_F0": 1, "CAM_R0": 2, "CAM_R1": 3,
            "CAM_R2": 4, "CAM_B0": 5, "CAM_L2": 6, "CAM_L1": 7
        }

        # 加载当前 ego→world
        ego_pose = db.ego_pose.get(pc.ego_pose_token)
        ego_T = np.eye(4)
        rot = Rotation.from_quat([ego_pose.qx, ego_pose.qy, ego_pose.qz, ego_pose.qw]).as_matrix()
        ego_T[:3, :3] = rot
        ego_T[:3, 3] = [ego_pose.x, ego_pose.y, ego_pose.z]

        # 遍历所有相机
        for cam in db.camera:
            cam_name = cam.channel
            if cam_name not in cam_name2id:
                continue
            cam_id = cam_name2id[cam_name]

            K = cam.intrinsic_np
            RT = cam.trans_matrix.T        # ego → cam
            RT[1, :] *= -1
            #RT = RT @ np.linalg.inv(ego_T)  # world → cam
            H, W = 1080, 1920

            depth_mask, depth_value = generate_depth_from_lidar(points_ego, K, RT, H, W)
            np.savez_compressed(os.path.join(depth_dir, f"{cam_idx:06d}_{cam_id}.npz"),
                                mask=depth_mask,
                                value=depth_value)

            img_path = os.path.join(save_dir, 'images', f'{cam_idx:06d}_{cam_id}.png')
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
            cv2.imwrite(os.path.join(depth_dir, f"{cam_idx:06d}_{cam_id}.png"), overlay)

# ---------------- entry ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuplan_root', required=True)
    parser.add_argument('--log_name',   required=True)
    parser.add_argument('--save_dir',   required=True)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir,args.log_name)
    export_lidar_pcd(args.nuplan_root, args.log_name, save_dir, args.skip_existing)
