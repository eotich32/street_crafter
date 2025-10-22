#!/usr/bin/env python3
# once_point_conventer.py
import os
import cv2
import numpy as np
import pickle
import argparse
from tqdm import tqdm

# ---------- 工具函数 ----------
def load_bin(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]                          # x,y,z

def load_pose(pose_path):
    return np.loadtxt(pose_path).reshape(4, 4)

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def load_KRT(extr_path, intr_path):
    extr = np.loadtxt(extr_path).reshape(4, 4)   # lidar -> cam
    intr = np.loadtxt(intr_path)
    K = intr[:3, :3]
    D = intr[3:] if intr.shape[0] >= 4 else np.zeros(5)
    return K, D, extr

def project_numpy(xyz, K, RT, h, w, dist_coeffs=None):
    """把 XYZ 投影到图像，返回 uv、mask"""
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    rvec, _ = cv2.Rodrigues(RT[:3, :3])
    tvec = RT[:3, 3]
    pts, _ = cv2.projectPoints(xyz, rvec, tvec, K, dist_coeffs)
    pts = pts.reshape(-1, 2)
    mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    return pts, mask

def bbox_to_corner3d(bbox):
    """bbox: [x,y,z,l,w,h,yaw] -> 8x3 corners"""
    x, y, z, l, w, h, yaw = bbox
    cos, sin = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [-l/2, -w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])
    R = np.array([[cos, -sin, 0],
                  [sin,  cos, 0],
                  [0,    0,   1]])
    corners = (R @ corners.T).T + np.array([x, y, z])
    return corners

def inbbox_points(xyz, corners):
    """判断点是否在 3D bbox 内（简易 AABB）"""
    min_c = corners.min(0)
    max_c = corners.max(0)
    return ((xyz >= min_c) & (xyz <= max_c)).all(1)

def storePly(path, xyz, rgb):
    """极简 ply 写入（ASCII）"""
    N = xyz.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} "
                    f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])}\n")

def visualize_depth_numpy(depth, max_depth=80):
    """深度图 -> 伪彩色"""
    depth = np.clip(depth, 0, max_depth)
    gray = (depth / max_depth * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)
    return color[..., ::-1], gray

# ---------- 主流程 ----------
def process(seq_dir, seq_name):
    base = os.path.join(seq_dir, seq_name)
    for sub in ['images', 'lidar', 'ego_pose', 'extrinsics', 'intrinsics', 'track']:
        assert os.path.exists(os.path.join(base, sub)), f"Missing {sub}"

    img_dir      = os.path.join(base, 'images')
    lidar_in_dir = os.path.join(base, 'lidar')
    ego_dir      = os.path.join(base, 'ego_pose')
    extr_dir     = os.path.join(base, 'extrinsics')
    intr_dir     = os.path.join(base, 'intrinsics')
    track_dir    = os.path.join(base, 'track')

    out_bg   = os.path.join(base, 'lidar', 'background'); os.makedirs(out_bg, exist_ok=True)
    out_actor= os.path.join(base, 'lidar', 'actor');      os.makedirs(out_actor, exist_ok=True)
    out_depth= os.path.join(base, 'lidar', 'depth');      os.makedirs(out_depth, exist_ok=True)

    track_info = load_pkl(os.path.join(track_dir, 'track_info.pkl'))

    # 放在 track_info = load_pkl(...) 之后
    print('track_info keys (前10个):', list(track_info.keys())[:1000])
    print('类型示例:', type(list(track_info.keys())[0]))

    bin_files = sorted([f for f in os.listdir(lidar_in_dir) if f.endswith('.bin')])
    frame_ids = [f.split('.')[0] for f in bin_files]

    # 预加载所有相机标定
    KRT = {}
    for cam_id in range(8):
        extr_path = os.path.join(extr_dir, f'{cam_id}.txt')
        intr_path = os.path.join(intr_dir, f'{cam_id}.txt')
        if os.path.exists(extr_path) and os.path.exists(intr_path):
            K, D, extr = load_KRT(extr_path, intr_path)
            KRT[cam_id] = {'K': K, 'D': D, 'extr': extr}

    for frame_id in tqdm(frame_ids, desc=seq_name):
        if int(frame_id) not in track_info:
            #print(f"[WARN] frame {int(frame_id)} not in track_info")
            continue
        xyz = load_bin(os.path.join(lidar_in_dir, f'{frame_id}.bin'))
        rgb = np.zeros_like(xyz, dtype=np.uint8)

        # 投影 + 着色 + 深度图
        for cam_id, calib in KRT.items():
            img_path = os.path.join(img_dir, f'{int(frame_id):06d}_{cam_id}.png')
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)[..., ::-1]
            h, w = img.shape[:2]
            K, D, extr = calib['K'], calib['D'], calib['extr']
            uv, mask = project_numpy(xyz, K, extr, h, w, D)
            u, v = uv[mask][:, 0].astype(np.int32), uv[mask][:, 1].astype(np.int32)
            rgb[mask] = img[v, u]

            # 深度图
            depth = np.ones((h, w), dtype=np.float32) * 1e5
            xyz_cam = (extr @ np.hstack([xyz[mask], np.ones((mask.sum(), 1))]).T).T[:, :3]
            z = xyz_cam[:, 2]
            depth[v, u] = np.minimum(depth[v, u], z)
            depth[depth >= 1e5] = 0
            np.savez_compressed(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.npz'),
                                mask=depth > 0, value=depth[depth > 0])
            depth_vis, _ = visualize_depth_numpy(depth)
            depth_vis_img = img.copy()
            depth_vis_img[depth > 0] = depth_vis[depth > 0]
            cv2.imwrite(os.path.join(out_depth, f'{int(frame_id):06d}_{cam_id}.png'), depth_vis_img[..., ::-1])

        # 背景点云
        storePly(os.path.join(out_bg, f'{int(frame_id):06d}.ply'), xyz, rgb)

        # actor 点云
        if int(frame_id) in track_info:
            for obj in track_info[int(frame_id)]:
                track_id = obj['track_id']
                bbox = np.array(obj['bbox'])  # [x,y,z,l,w,h,yaw]
                T_v2l = np.linalg.inv(load_pose(os.path.join(ego_dir, f'{frame_id}.txt')))  # vehicle->lidar
                bbox_pos = np.array([bbox[0], bbox[1], bbox[2], 1.0])
                bbox[:3] = (T_v2l @ bbox_pos)[:3]
                corners = bbox_to_corner3d(bbox)
                mask = inbbox_points(xyz, corners)
                if mask.sum() == 0:
                    continue
                actor_sub = os.path.join(out_actor, track_id)
                os.makedirs(actor_sub, exist_ok=True)
                storePly(os.path.join(actor_sub, f'{int(frame_id):06d}.ply'), xyz[mask], rgb[mask])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', required=True, help='path to processed/')
    parser.add_argument('--seq_name', required=True, help='e.g. 000027')
    args = parser.parse_args()
    process(args.seq_dir, args.seq_name)

if __name__ == '__main__':
    main()