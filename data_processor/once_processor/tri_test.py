
import os
import cv2
import numpy as np
import pickle
import argparse
from scipy.spatial.transform import Rotation as R

# ---------- 离线 3D 渲染 ----------
import open3d as o3d


def save_3d_offline(pts, trajectory, visible_tids, path):
    """pts: (N,3) ndarray; visible_tids: list[tid]"""
    # 1. 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 2. 材质
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # 3. 离线渲染器
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    renderer.scene.set_background([0, 0, 0, 1])          # 黑色背景
    renderer.scene.add_geometry("pcd", pcd, mat)         # 必须给名字

    # 4. 3D 框
    for i, tid in enumerate(visible_tids):
        traj = trajectory[tid]
        pose = traj['poses_vehicle'][0]
        center = pose[:3, 3]
        l, wh, ht = traj['length'], traj['width'], traj['height']
        obb = o3d.geometry.OrientedBoundingBox()
        obb.center, obb.extent, obb.R = center, [l, wh, ht], pose[:3, :3]
        obb.color = [1, 0, 0]
        print('pose[:3, :3] =\n', pose[:3, :3])
        renderer.scene.add_geometry(f"box_{i}", obb, mat)

    # 5. 相机参数 & 渲染
    renderer.setup_camera(60, [0, 0, 0], [30, 0, 10], [0, 0, 1])
    img_o3d = renderer.render_to_image()
    o3d.io.write_image(path, img_o3d)
    print(f'[INFO] 已保存 3D 图（离线渲染）：{path}')


# ---------- 参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description='Debug track (headless).')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0, help='frame index')
    parser.add_argument('--cam', type=int, default=0, help='camera index')
    return parser.parse_args()


# ---------- 主逻辑 ----------
def main():
    args = parse_args()
    out, idx, cam_i = args.output_dir, args.idx, args.cam
    fid = f'{idx:06d}'
    debug_dir = f'{out}/debug_vis'
    os.makedirs(debug_dir, exist_ok=True)

    # 公共加载
    with open(f'{out}/track/trajectory.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    with open(f'{out}/track/track_camera_visible.pkl', 'rb') as f:
        track_cam_vis = pickle.load(f)

    intr = np.loadtxt(f'{out}/intrinsics/{cam_i}.txt')
    extr = np.loadtxt(f'{out}/extrinsics/{cam_i}.txt')
    extr = np.linalg.inv(extr)

    img_path = f'{out}/images/{fid}_{cam_i}.jpg'
    lidar_path = f'{out}/lidar/{fid}.bin'
    mask_path = f'{out}/dynamic_mask/{fid}_{cam_i}.jpg'

    # 1. 3D 可视化（离线）
    if os.path.exists(lidar_path):
        pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        visible_tids = [tid for tid in track_cam_vis.get(fid, {}).get(cam_i, [])
                        if tid in trajectory]
        save_3d_offline(pts, trajectory, visible_tids,
                        f'{debug_dir}/3d_{fid}_{cam_i}.png')
    else:
        print(f'[WARN] 未找到 lidar/{fid}.bin，跳过 3D 可视化')

    # 2. 2D 投影框
    if os.path.exists(img_path):
        img_bgr = cv2.imread(img_path)
        img2 = img_bgr.copy()
        for tid in track_cam_vis.get(fid, {}).get(cam_i, []):
            if tid not in trajectory:
                continue
            traj = trajectory[tid]
            pose = traj['poses_vehicle'][0]
            l, wh, ht = traj['length'], traj['width'], traj['height']
            corners = np.array([
                [-l/2, -wh/2, -ht/2], [l/2, -wh/2, -ht/2],
                [l/2, wh/2, -ht/2], [-l/2, wh/2, -ht/2],
                [-l/2, -wh/2, ht/2], [l/2, -wh/2, ht/2],
                [l/2, wh/2, ht/2], [-l/2, wh/2, ht/2],
            ])
            corners = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
            corners_cam = (extr[:3, :3] @ corners.T + extr[:3, 3:4]).T
            pts_2d = (intr @ corners_cam.T).T
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
            pts_2d = pts_2d.astype(int)
            bottom = pts_2d[:4]
            cv2.polylines(img2, [bottom], isClosed=True,
                          color=(0, 255, 0), thickness=2)
        cv2.imwrite(f'{debug_dir}/2d_proj_{fid}_{cam_i}.png', img2)
        print(f'[INFO] 已保存 2D 投影图：{debug_dir}/2d_proj_{fid}_{cam_i}.png')
    else:
        print(f'[WARN] 未找到 images/{fid}_{cam_i}.jpg，跳过 2D 投影')

    # 3. mask 叠加
    if os.path.exists(mask_path) and os.path.exists(img_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        overlay = img_bgr.copy()
        overlay[mask > 0] = [0, 0, 255]
        vis_mask = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        cv2.imwrite(f'{debug_dir}/mask_overlay_{fid}_{cam_i}.png', vis_mask)
        print(f'[INFO] 已保存 mask 叠加图：{debug_dir}/mask_overlay_{fid}_{cam_i}.png')
    else:
        print(f'[WARN] 缺少图像或掩膜，跳过 mask 叠加')

    print(f'✅ 调试完成，结果保存在 {debug_dir}/')


if __name__ == '__main__':
    main()
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整 8 角点调试版
Usage:
    python debug_track_full8.py --output_dir /path/to/processed/000027 --idx 10 --cam 0
"""
import os
import cv2
import numpy as np
import pickle
import argparse
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description='Debug 8 corners')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0, help='frame index')
    parser.add_argument('--cam', type=int, default=0, help='camera index')
    return parser.parse_args()


def main():
    args = parse_args()
    out, idx, cam_i = args.output_dir, args.idx, args.cam
    fid = f'{idx:06d}'
    debug_dir = f'{out}/debug_vis'
    os.makedirs(debug_dir, exist_ok=True)

    # 加载
    with open(f'{out}/track/trajectory.pkl', 'rb') as f:
        trajectory = pickle.load(f)
    with open(f'{out}/track/track_camera_visible.pkl', 'rb') as f:
        track_cam_vis = pickle.load(f)

    intr = np.loadtxt(f'{out}/intrinsics/{cam_i}.txt')
    extr = np.loadtxt(f'{out}/extrinsics/{cam_i}.txt')
    extr = np.linalg.inv(extr)  # velo -> cam

    img_path = f'{out}/images/{fid}_{cam_i}.jpg'
    if not os.path.exists(img_path):
        print('[WARN] 无图像')
        return

    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    img_out = img_bgr.copy()

    for tid in track_cam_vis.get(fid, {}).get(cam_i, []):
        if tid not in trajectory:
            continue
        traj = trajectory[tid]
        pose = traj['poses_vehicle'][0]
        l, wh, ht = traj['length'], traj['width'], traj['height']

        # 8 角点（物体坐标系）
        corners = np.array([
            [-l/2, -wh/2, -ht/2], [l/2, -wh/2, -ht/2],
            [l/2, wh/2, -ht/2], [-l/2, wh/2, -ht/2],
            [-l/2, -wh/2, ht/2], [l/2, -wh/2, ht/2],
            [l/2, wh/2, ht/2], [-l/2, wh/2, ht/2],
        ])
        corners = (pose[:3, :3] @ corners.T).T + pose[:3, 3]
        corners_cam = (extr[:3, :3] @ corners.T + extr[:3, 3:4]).T
        pts_2d = (intr @ corners_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
        pts_2d = pts_2d.astype(int)

        # 1. 画 8 点 + 编号
        for idx_pt, (x, y) in enumerate(pts_2d):
            cv2.circle(img_out, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(img_out, str(idx_pt), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 2. 画底面 4 边（绿色）
        #bottom = pts_2d[:4]
        bottom = pts_2d[[0, 1, 2, 3]][::-1]
        # 底面绿色
        cv2.polylines(img_out, [bottom], True, (0, 255, 0), 2)

        # 3. 画前面 4 边（红色）→ 0-1-5-4
        front = pts_2d[[0, 1, 5, 4]]
        # 前面红色
        cv2.polylines(img_out, [front], True, (0, 0, 255), 2)

    out_path = f'{debug_dir}/2d_proj_{fid}_{cam_i}_FULL8.png'
    cv2.imwrite(out_path, img_out)
    print(f'[INFO] 已保存 8 角点调试图：{out_path}')


if __name__ == '__main__':
    main()
'''