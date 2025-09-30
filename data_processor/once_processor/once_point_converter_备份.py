
#!/usr/bin/env python3
# once_point_converter.py → 采用 lidar2image 同款投影
import os
import cv2
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import open3d as o3d

# ---------- 工具 ----------
def load_bin(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]

def voxel_downsample_indices(xyz: np.ndarray,
                             voxel_size: float) -> np.ndarray:
    """
    纯 numpy 实现：对 N×3 点云做体素降采样，返回保留的原始索引。
    规则：每个 voxel 只保留第一个落进来的点。

    Parameters
    ----------
    xyz : np.ndarray, shape=(N, 3), dtype=float32/float64
        原始点云坐标
    voxel_size : float
        体素边长

    Returns
    -------
    np.ndarray, shape=(M,), dtype=int64
        被选中的原始点索引，M ≤ N
    """
    # 1. 计算每个点的 voxel 坐标（整数 key）
    voxel_coord = np.floor(xyz / voxel_size).astype(np.int32)

    # 2. 用字典记录每个 voxel 第一次出现的索引
    voxel_dict = {}
    for i, vox in enumerate(voxel_coord):
        key = (vox[0], vox[1], vox[2])
        if key not in voxel_dict:   # 只保留第一个
            voxel_dict[key] = i

    # 3. 返回索引数组
    return np.fromiter(voxel_dict.values(), dtype=np.int64)

# ---------- 统一内参读取（与 once_converter.py 保持一致） ----------
def read_intrinsic(cam_path: str):
    """
    cam_path: /xxx/xxx/intrinsics/{cam_id}.txt
    文件内容: fx fy cx cy k1 k2 p1 p2 k3   （9 个数）
    返回 K(3×3), dist(5,)
    """
    vec = np.loadtxt(cam_path)
    if vec.ndim == 0:          # 单元素退化成 0-d
        vec = vec.reshape(-1)
    if vec.size != 9:
        raise ValueError(
            f'内参格式错误！期望 9 个数 (fx fy cx cy k1 k2 p1 p2 k3)，'
            f'实际得到 {vec.size} 个：{vec}'
        )
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = vec
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    dist = np.array([k1, k2, p1, p2, k3])
    return K, dist                    

def load_KRT(extr_path, intr_path):
    extr = np.loadtxt(extr_path).reshape(4, 4)   # lidar->cam
    K, dist = read_intrinsic(intr_path)          # K 已经是 3×3，dist 是 5
    return K, dist, extr

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
'''
def load_KRT(extr_path, intr_path):
    extr = np.loadtxt(extr_path).reshape(4, 4)   # lidar->cam
    intr = np.loadtxt(intr_path)
    K = intr[:3, :3]
    D = intr[3:] if intr.shape[0] >= 4 else np.zeros(5)
    return K, D, extr
'''
def bbox_to_corner3d(bbox):
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

def inbbox_points(xyz, corners):
    mn = corners.min(0); mx = corners.max(0)
    return ((xyz >= mn) & (xyz <= mx)).all(1)

def store_ply(path, xyz, rgb,mask=None):
    N = xyz.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if mask is not None:
            f.write("property uchar mask\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} "
                    f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])}")
            if mask is not None:
                f.write(f" {int(mask[i])}")
            f.write("\n")
'''
def downsample_and_save(xyz, rgb, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.15)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    xyz_out = np.asarray(pcd.points).astype(np.float32)
    rgb_out = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    store_ply(save_path, xyz_out, rgb_out)
'''
def downsample_and_save(xyz, rgb, save_path):
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors

    # ① 只拿索引，不改坐标
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    if hasattr(o3d.geometry.PointCloud, 'get_voxel_down_sample_indices'):
        idx = np.asarray(pcd.get_voxel_down_sample_indices())
    else:
        idx = voxel_downsample_indices(xyz, 0.15)

    xyz_out = xyz[idx]      # ➜ 用原始坐标
    rgb_out = rgb[idx]

    # ② 半径离群
    neigh_idx = nn.radius_neighbors(xyz_out, return_distance=False)
    neighbors = np.array([len(idx) for idx in neigh_idx])
    keep = neighbors > 10
    xyz_out = xyz_out[keep]
    rgb_out = rgb_out[keep]

    # ③ 写盘
    store_ply(save_path, xyz_out, rgb_out)

# ---------- 新增：lidar2image 同款投影 ----------
IMAGE_SIZE = (1920, 1020)   # (W, H)

def project_lidar_to_image(pts, T_velo2cam, K, img_size=(1920, 1020)):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
             (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    
    # 返回原始点云长度的掩码和对应的 uv
    full_mask = np.zeros(pts.shape[0], dtype=bool)
    full_mask[mask] = inside
    uv_full = np.zeros((pts.shape[0], 2), dtype=int)
    uv_full[full_mask] = pts_2d[inside].astype(int)
    z_full = np.zeros(pts.shape[0], dtype=float)
    z_full[full_mask] = pts_cam[inside, 2]
    
    return uv_full, full_mask, z_full

# ---------- 主流程 ----------
# ---------- 主流程（已过滤单帧 actor） ----------
'''
def process(seq_dir, seq_name, CAM_IDS):
    base = os.path.join(seq_dir, seq_name)
    for sub in ['images', 'lidar', 'ego_pose', 'extrinsics', 'intrinsics', 'track']:
        assert os.path.exists(os.path.join(base, sub)), f"Missing {sub}"

    img_dir      = os.path.join(base, 'images')
    lidar_in_dir = os.path.join(base, 'lidar')
    track_dir    = os.path.join(base, 'track')

    out_bg   = os.path.join(base, 'lidar', 'background'); os.makedirs(out_bg, exist_ok=True)
    out_actor= os.path.join(base, 'lidar', 'actor');      os.makedirs(out_actor, exist_ok=True)
    out_depth= os.path.join(base, 'lidar', 'depth');      os.makedirs(out_depth, exist_ok=True)

    track_info = load_pkl(os.path.join(track_dir, 'track_info.pkl'))
    track_info = {int(k): v for k, v in track_info.items()}

    bin_files = sorted([f for f in os.listdir(lidar_in_dir) if f.endswith('.bin')])
    frame_ids = [f.split('.')[0] for f in bin_files]

    KRT = {}
    for cam_id in CAM_IDS:
        K, D, extr = load_KRT(os.path.join(base, 'extrinsics', f'{cam_id}.txt'),
                              os.path.join(base, 'intrinsics', f'{cam_id}.txt'))
        KRT[cam_id] = {'K': K, 'D': D, 'extr': extr}

    bg_xyz_list, bg_rgb_list = [], []
    actor_dict = {}   # tid -> {'xyz': [], 'rgb': [], 'frame_list': []}  # 🔧

    # ---------- 先收集所有帧 ----------
    for frame_id in tqdm(frame_ids, desc=seq_name):
        xyz = load_bin(os.path.join(lidar_in_dir, f'{frame_id}.bin'))
        rgb = np.zeros_like(xyz, dtype=np.uint8)
        mask = np.zeros(xyz.shape[0], dtype=bool)

        for cam_id, calib in KRT.items():
            img_path = os.path.join(img_dir, f'{int(frame_id):06d}_{cam_id}.jpg')
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)[..., ::-1]
            h, w = img.shape[:2]
            K, extr = calib['K'], calib['extr']

            uv, inlier, z = project_lidar_to_image(xyz, np.linalg.inv(extr), K, (w, h))
            if inlier.any():
                rgb[inlier] = img[uv[inlier, 1], uv[inlier, 0]]
                mask[inlier] = True

            # ---------- 深度图 ----------
            depth = np.ones((h, w), dtype=np.float32) * 1e5
            if len(uv) > 0:
                depth[uv[:, 1], uv[:, 0]] = np.minimum(depth[uv[:, 1], uv[:, 0]], z)
            depth[depth >= 1e5] = 0
            np.savez_compressed(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.npz'),
                                mask=depth > 0, value=depth[depth > 0])
            depth_vis = cv2.applyColorMap((np.clip(depth, 0, 80) / 80 * 255).astype(np.uint8),
                                          cv2.COLORMAP_PLASMA)
            depth_vis_img = img.copy()
            depth_vis_img[depth > 0] = depth_vis[depth > 0]
            cv2.imwrite(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.jpg'),
                        depth_vis_img[..., ::-1])

        # ---------- 保存背景 ----------
        store_ply(os.path.join(out_bg, f'{int(frame_id):06d}.ply'), xyz, rgb, mask)
        bg_xyz_list.append(xyz)
        bg_rgb_list.append(rgb)

        # ---------- 抠 actor ----------
        frame_idx = int(frame_id)
        if frame_idx in track_info:
            for tid, obj in track_info[frame_idx].items():
                bbox = np.array([obj['lidar_box'][k] for k in
                                 ['center_x', 'center_y', 'center_z',
                                  'length', 'width', 'height', 'heading']])
                corners = bbox_to_corner3d(bbox)
                mask_actor = inbbox_points(xyz, corners)
                if mask_actor.sum() == 0:
                    continue
                actor_sub = os.path.join(out_actor, str(tid))
                os.makedirs(actor_sub, exist_ok=True)
                store_ply(os.path.join(actor_sub, f'{int(frame_id):06d}.ply'),
                          xyz[mask_actor], rgb[mask_actor], mask_actor)
                # 记录帧号
                if tid not in actor_dict:
                    actor_dict[tid] = {'xyz': [], 'rgb': [], 'frame_list': []}
                actor_dict[tid]['xyz'].append(xyz[mask_actor])
                actor_dict[tid]['rgb'].append(rgb[mask_actor])
                actor_dict[tid]['frame_list'].append(frame_idx)

    # ---------- 过滤单帧 actor ----------
    # 只保留出现多帧的 track
    actor_dict = {tid: data for tid, data in actor_dict.items()
                  if len(set(data['frame_list'])) > 1}

    # ---------- 输出 full.ply ----------
    if bg_xyz_list:
        bg_xyz = np.concatenate(bg_xyz_list, axis=0)
        bg_rgb = np.concatenate(bg_rgb_list, axis=0)
        downsample_and_save(bg_xyz, bg_rgb, os.path.join(out_bg, 'full.ply'))
    for tid, data in actor_dict.items():
        xyz = np.concatenate(data['xyz'], axis=0)
        rgb = np.concatenate(data['rgb'], axis=0)
        downsample_and_save(xyz, rgb, os.path.join(out_actor, tid, 'full.ply'))
'''
# ---------- 主流程（与 waymo 保存逻辑对齐） ----------
def process(seq_dir, seq_name, CAM_IDS):
    base = os.path.join(seq_dir, seq_name)
    for sub in ['images', 'lidar', 'ego_pose', 'extrinsics', 'intrinsics', 'track']:
        assert os.path.exists(os.path.join(base, sub)), f"Missing {sub}"

    img_dir      = os.path.join(base, 'images')
    lidar_in_dir = os.path.join(base, 'lidar')
    track_dir    = os.path.join(base, 'track')

    out_bg   = os.path.join(base, 'lidar', 'background'); os.makedirs(out_bg, exist_ok=True)
    out_actor= os.path.join(base, 'lidar', 'actor');      os.makedirs(out_actor, exist_ok=True)
    out_depth= os.path.join(base, 'lidar', 'depth');      os.makedirs(out_depth, exist_ok=True)

    #track_info = load_pkl(os.path.join(track_dir, 'track_info.pkl'))
    #track_info = {int(k): v for k, v in track_info.items()}
    track_info = load_pkl(os.path.join(track_dir, 'track_info.pkl'))
    track_info = {int(k): v for k, v in track_info.items()}
    # ↓↓↓ 加载 trajectory
    trajectory = load_pkl(os.path.join(track_dir, 'trajectory.pkl'))

    bin_files = sorted([f for f in os.listdir(lidar_in_dir) if f.endswith('.bin')])
    frame_ids = [f.split('.')[0] for f in bin_files]

    KRT = {}
    for cam_id in CAM_IDS:
        K, D, extr = load_KRT(os.path.join(base, 'extrinsics', f'{cam_id}.txt'),
                              os.path.join(base, 'intrinsics', f'{cam_id}.txt'))
        KRT[cam_id] = {'K': K, 'D': D, 'extr': extr}

    # 只用来合并 actor full.ply
    actor_dict = {}

    for frame_id in tqdm(frame_ids, desc=seq_name):
        xyz = load_bin(os.path.join(lidar_in_dir, f'{frame_id}.bin'))
        rgb = np.zeros_like(xyz, dtype=np.uint8)
        mask = np.zeros(xyz.shape[0], dtype=bool)

        # ---------- 投影染色 ----------
        for cam_id, calib in KRT.items():
            img_path = os.path.join(img_dir, f'{int(frame_id):06d}_{cam_id}.jpg')
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)[..., ::-1]
            h, w = img.shape[:2]
            K, extr = calib['K'], calib['extr']

            uv, inlier, z = project_lidar_to_image(xyz, np.linalg.inv(extr), K, (w, h))
            if inlier.any():
                rgb[inlier] = img[uv[inlier, 1], uv[inlier, 0]]
                mask[inlier] = True

            # 深度图（略，与你原代码相同）
            depth = np.ones((h, w), dtype=np.float32) * 1e5
            if len(uv) > 0:
                depth[uv[:, 1], uv[:, 0]] = np.minimum(depth[uv[:, 1], uv[:, 0]], z)
            depth[depth >= 1e5] = 0
            np.savez_compressed(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.npz'),
                                mask=depth > 0, value=depth[depth > 0])
            #depth_vis = cv2.applyColorMap((np.clip(depth, 0, 80) / 80 * 255).astype(np.uint8),
                                          #cv2.COLORMAP_PLASMA)
            depth_norm = np.clip(depth, 0, 80) / 80
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_JET) 
            
            depth_vis_img = img.copy()
            depth_vis_img[depth > 0] = depth_vis[depth > 0]
            cv2.imwrite(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.jpg'),
                        depth_vis_img[..., ::-1])

        # ---------- 背景：单帧直接写原始坐标 ----------
        store_ply(os.path.join(out_bg, f'{int(frame_id):06d}.ply'), xyz, rgb, mask)

        frame_idx = int(frame_id)
        if frame_idx in track_info:
            for tid, obj in track_info[frame_idx].items():
                bbox = np.array([obj['lidar_box'][k] for k in
                                ['center_x', 'center_y', 'center_z',
                                'length', 'width', 'height', 'heading']])
                corners = bbox_to_corner3d(bbox)
                #bbox[3:6] *= 1.2 
                inbbox_mask = inbbox_points(xyz, corners)   # 世界系下抠点
                if inbbox_mask.sum() == 0:
                    continue
                actor_sub = os.path.join(out_actor, str(tid))
                os.makedirs(actor_sub, exist_ok=True)

                # ① 世界 → 局部框（与 full.ply 一致）
                pose_v = trajectory[tid]['poses_vehicle'][trajectory[tid]['frames'].index(frame_idx)]
                xyz_w = xyz[inbbox_mask]
                xyz_h = np.concatenate([xyz_w, np.ones((xyz_w.shape[0], 1))], axis=1)
                xyz_l = (np.linalg.inv(pose_v) @ xyz_h.T).T[:, :3]
                rgb_l = rgb[inbbox_mask]
                mask_uint8 = inbbox_mask.astype(np.uint8) 

                # ② 写局部框坐标
                store_ply(os.path.join(actor_sub, f'{int(frame_id):06d}.ply'), xyz_l, rgb_l,mask_uint8)

                # ③ 累积到合并池（也是局部框）
                if tid not in actor_dict:
                    #actor_dict[tid] = {'xyz': [], 'rgb': []}
                    actor_dict[tid] = {'xyz': [], 'rgb': [], 'mask': []} 
                actor_dict[tid]['xyz'].append(xyz_l)
                actor_dict[tid]['rgb'].append(rgb_l)
                actor_dict[tid]['mask'].append(mask_uint8)

    # ---------- 仅合并 actor full.ply（与 waymo 脚本一致） ----------
    '''
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors
    for tid, data in actor_dict.items():
        xyz_world_list = data['xyz']          # List[N,3] 世界坐标
        rgb_list       = data['rgb']
        pose_list      = [trajectory[tid]['poses_vehicle'][i]
                          for i in range(len(xyz_world_list))]  # 对应帧的 pose

        # ① 世界 → 局部框（与 waymo 一致）
        xyz_local_list = []
        rgb_local_list = []
        for xyz_w, rgb_f, pose_v in zip(xyz_world_list, rgb_list, pose_list):
            # pose_v: 世界→vehicle，局部框 = inv(pose_v)
            xyz_h = np.concatenate([xyz_w, np.ones((xyz_w.shape[0], 1))], axis=1)
            xyz_l = (np.linalg.inv(pose_v) @ xyz_h.T).T[:, :3]
            xyz_local_list.append(xyz_l)
            rgb_local_list.append(rgb_f)

        xyz_local = np.concatenate(xyz_local_list, axis=0)
        rgb_local = np.concatenate(rgb_local_list, axis=0)

        # ② 局部框下合并 + 降采样（索引法）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_local)
        pcd.colors = o3d.utility.Vector3dVector(rgb_local / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.15)
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.5)

        # 拿索引回原始数组
        if hasattr(o3d.geometry.PointCloud, 'get_voxel_down_sample_indices'):
            idx = np.asarray(pcd.get_voxel_down_sample_indices())
        else:
            idx = voxel_downsample_indices(xyz_local, 0.15)
        xyz_out = xyz_local[idx]
        rgb_out = rgb_local[idx]

        store_ply(os.path.join(out_actor, tid, 'full.ply'), xyz_out, rgb_out)
    '''
    # ---------- 仅合并 actor full.ply（局部框坐标，不再变换） ----------
    import open3d as o3d

    for tid, data in actor_dict.items():
        if tid not in trajectory:
            continue
        xyz_local = np.concatenate(data['xyz'], axis=0)   # 已是局部框
        rgb_local = np.concatenate(data['rgb'], axis=0)
        mask_local = np.concatenate(data['mask'], axis=0)

        # 仅做降采样（索引法）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_local)
        pcd.colors = o3d.utility.Vector3dVector(rgb_local / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=0.15)
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.5)

        # 拿索引回原始数组
        if hasattr(o3d.geometry.PointCloud, 'get_voxel_down_sample_indices'):
            idx = np.asarray(pcd.get_voxel_down_sample_indices())
        else:
            idx = voxel_downsample_indices(xyz_local, 0.15)
        xyz_out = xyz_local[idx]
        rgb_out = rgb_local[idx]
        mask_out = mask_local[idx]

        # 写局部框 + mask
        store_ply(os.path.join(out_actor, tid, 'full.ply'), xyz_out, rgb_out, mask_out)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', required=True, help='path to processed/')
    parser.add_argument('--seq_name', required=True, help='e.g. 000027')
    parser.add_argument('--cams', type=str, default=None,
                        help='要处理的相机，英文逗号分隔，如“0,1,2”代表 cam00~cam02；'
                             '留空则默认处理 0~6 共 7 个相机')
    args = parser.parse_args()

    ALL_CAMERAS = list(range(7))          # 默认 0~6
    if args.cams is None:
        CAM_IDS = ALL_CAMERAS
    else:
        # 去重 + 排序 + 合法性检查
        try:
            CAM_IDS = sorted({int(c.strip()) for c in args.cams.split(',')})
        except ValueError:
            raise ValueError('--cams 必须是用逗号分隔的整数（如 0,1,5）')
        if not set(CAM_IDS).issubset(ALL_CAMERAS):
            raise ValueError(f'--cams 超出合法范围 {ALL_CAMERAS}')
    print('本次处理的相机 ID：', CAM_IDS)
    process(args.seq_dir, args.seq_name,CAM_IDS)

if __name__ == '__main__':
    main()