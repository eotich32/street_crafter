#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
import os
from pathlib import Path

IMAGE_SIZE = (1920, 1020)  # (W, H)

def load_bin(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def load_txt(txt_path):
    return np.loadtxt(txt_path)

def project_lidar_to_image(pts, T_velo2cam, K, img_size=IMAGE_SIZE):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    h, w = img_size[1], img_size[0]
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
             (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int)

def main():
    parser = argparse.ArgumentParser(description='LiDAR -> Image 全帧可视化')
    parser.add_argument('data_dir', type=str, help='ONCE 处理根目录')
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / 'lidar_check'
    out_dir.mkdir(exist_ok=True)

    # 自动统计帧数：按 images 目录下的第一张相机图
    img_dir = Path(args.data_dir) / 'images'
    all_imgs = sorted(img_dir.glob('*_0.jpg'))  # 用 cam0 图计数
    n_frames = len(all_imgs)
    print(f'共发现 {n_frames} 帧，开始逐帧投影...')

    for frame_idx in range(n_frames):
        for cam_idx in range(7):   # cam0..6
            img_path   = img_dir / f'{frame_idx:06d}_{cam_idx}.jpg'
            lidar_path = Path(args.data_dir) / 'lidar' / f'{frame_idx:06d}.bin'
            K_path = Path(args.data_dir) / 'intrinsics' / f'{cam_idx}.txt'
            T_path = Path(args.data_dir) / 'extrinsics' / f'{cam_idx}.txt'

            if not img_path.exists() or not lidar_path.exists():
                continue

            img   = cv2.imread(str(img_path))
            pts   = load_bin(str(lidar_path))
            K     = load_txt(str(K_path))
            T_velo2cam = np.linalg.inv(load_txt(str(T_path)))

            uv = project_lidar_to_image(pts, T_velo2cam, K)
            for (u, v) in uv:
                cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)

            out_path = out_dir / f'{frame_idx:06d}_{cam_idx}.jpg'
            cv2.imwrite(str(out_path), img)

        if frame_idx % 50 == 0:
            print(f'  progress {frame_idx}/{n_frames}')

    print(f'全部完成，结果保存在 {out_dir}')

if __name__ == '__main__':
    main()