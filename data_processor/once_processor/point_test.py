#!/usr/bin/env python3
import numpy as np
import cv2
import argparse
import pickle
from pathlib import Path

IMAGE_SIZE = (1920, 1020)

def load_txt(p):
    return np.loadtxt(p)

# 正式代码的 corners 生成（相机系）
def corners_cam(box_data, yaw):
    x, y, z, l, w, h = box_data[:6]
    y -= h / 2.0                      # 贴地
    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    corners = (rot @ corners.T).T + [x, y, z]
    return corners

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('frame_idx', type=int)
    parser.add_argument('cam_idx',  type=int)
    args = parser.parse_args()

    dd = Path(args.data_dir)
    img_path = dd / 'images'   / f'{args.frame_idx:06d}_{args.cam_idx}.jpg'
    K_path   = dd / 'intrinsics' / f'{args.cam_idx}.txt'
    T_path   = dd / 'extrinsics' / f'{args.cam_idx}.txt'
    info_path= dd / 'track' / 'track_info.pkl'

    img = cv2.imread(str(img_path))
    K   = load_txt(K_path)
    T_velo2cam = np.linalg.inv(load_txt(T_path))

    # 只读当前帧
    fid = f'{args.frame_idx:06d}'
    track_info = pickle.loads(info_path.read_bytes())
    if fid not in track_info or not track_info[fid]:
        print('本帧无 3D 框')
        return

    for tid, info in track_info[fid].items():
        box = info['lidar_box']
        # 正式代码的 box_data：y↔z 互换
        x, y, z = box['center_z'], -box['center_y'], box['center_x']
        l, w, h = box['length'], box['width'], box['height']
        yaw = 0.0
        box_data = [x, y, z, l, w, h]

        # ① 生成 8 角点（相机系）
        corners = corners_cam(box_data, yaw)

        # ② 只剔相机后方
        valid = corners[:, 2] > 0
        corners = corners[valid]
        if len(corners) < 8:
            continue   # 整个框在后方，跳过

        # ③ 投影
        # ③ 投影
        uv, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), K, None)
        uv = uv.reshape(-1, 2).astype(int)

        # ④ 只留相机前方（Z>0）
        z_front = corners[:, 2] > 0
        uv = uv[z_front]
        if len(uv) == 0:
            continue

        # ⑤ Clamp 到图像内
        h_img, w_img = img.shape[:2]
        uv[:, 0] = np.clip(uv[:, 0], 0, w_img - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, h_img - 1)

# ⑥ 画点
        for j, (u, v) in enumerate(uv):
            cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
            cv2.putText(img, str(j), (u+4, v-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ⑥ 画线框
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5],
        ]
        for f in faces:
            cv2.polylines(img, [uv[f]], isClosed=True, color=(0, 255, 0), thickness=1)

    out_path = f'{args.data_dir}/tmp/box_clip_{args.frame_idx:06d}_{args.cam_idx}.jpg'
    cv2.imwrite(out_path, img)
    print(f'saved -> {out_path}')

if __name__ == '__main__':
    main()