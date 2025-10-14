import os
import argparse
import json
import shutil
import math
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation
import cv2,imageio
from copy import deepcopy

import pickle
import json

import torch
import pandas as pd

import matplotlib.pyplot as plt

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB

# ---------- 工具 ----------
def makedirs(p):
    os.makedirs(p, exist_ok=True)

def _round_11_digits(ts_us):
    if ts_us == 0:
        return 0.0
    scale = 10 ** (11 - int(math.log10(abs(ts_us))))
    return round(ts_us / scale) * scale

# ---------- 去畸变 ----------
def undistort_image(img_path, K, D, dst_size=(1920, 1080)):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((dst_size[1], dst_size[0], 3), np.uint8), K

    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0, newImgSize=dst_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dst_size, cv2.CV_16SC2)
    dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return dst, new_K          # <-- 多返回一个 new_K



# ---------- 每相机 200 帧 + 时间对齐 ----------
def _build_per_camera_frames(db, frames_per_cam=200, start_idx=0):
    Image = db.image[0].__class__
    query = (db.session.query(Image)
                       .order_by(Image.timestamp)
                       .offset(start_idx)
                       .limit(8 * frames_per_cam))
    all_rows = query.all()

    cam_batches = defaultdict(list)
    for img in all_rows:
        parts = img.filename_jpg.split('/')
        if len(parts) < 3:
            continue
        cam_name = parts[1]
        aligned_ts = _round_11_digits(img.timestamp)
        if len(cam_batches[cam_name]) < frames_per_cam:
            cam_batches[cam_name].append((aligned_ts, img))

    result = []
    cam_id_map = {name: idx for idx, name in enumerate(cam_batches.keys())}
    for cam_name, ts_imgs in cam_batches.items():
        cam_id = cam_id_map[cam_name]
        for new_idx, (aligned_ts, img) in enumerate(ts_imgs):
            result.append((new_idx, img, aligned_ts, cam_id, img.filename_jpg))
    return result, cam_id_map



# ---------- 主流程 ----------
def main(nuplan_root, log_name, save_dir, skip_existing=False):

    save_dir = os.path.join(save_dir, log_name)
    makedirs(save_dir)

    cam_newK_dict = {} 

    ego_dir = os.path.join(save_dir, 'ego_pose'); makedirs(ego_dir)
    intr_dir = os.path.join(save_dir, 'intrinsics'); makedirs(intr_dir)
    ext_dir = os.path.join(save_dir, 'extrinsics'); makedirs(ext_dir)
    img_dir = os.path.join(save_dir, 'images'); makedirs(img_dir)
    track_dir = os.path.join(save_dir, 'track'); makedirs(track_dir)
    dyn_dir = os.path.join(save_dir, 'dynamic_mask'); makedirs(dyn_dir)

    db_path = os.path.join(nuplan_root, 'nuplan-v1.1', 'splits', 'mini')

    print("[INFO] 初始化 NuPlanDB ...")
    db = NuPlanDB(data_root=db_path, load_path=log_name)

    #cam_id_map = _build_cam_id_map(db)
    #print("[INFO] 相机通道映射:", cam_id_map)

    ts_slice, cam_id_map = _build_per_camera_frames(db, frames_per_cam=100, start_idx=0)
    print("[INFO] 相机通道映射:", cam_id_map)

    # timestamps.json （相机名→{帧号:对齐时间}）
    ts_nested = defaultdict(dict)
    for idx, _, aligned_ts, cam_id, _ in ts_slice:
        cam_name = [k for k, v in cam_id_map.items() if v == cam_id][0]
        ts_nested[cam_name][f"{idx:06d}"] = float(aligned_ts)

    ts_json_path = os.path.join(save_dir, 'timestamps.json')
    with open(ts_json_path, 'w') as f:
        json.dump(ts_nested, f, indent=2)
    print(f"[INFO] timestamps.json 已写入，共 {len(ts_slice)} 条")

    ego_cache = {}
    for ego in db.ego_pose:
        aligned_ts = _round_11_digits(ego.timestamp)

        # 平移
        t = np.array([ego.x, ego.y, ego.z])
        # 四元数 → 旋转矩阵
        rot = Rotation.from_quat([ego.qx, ego.qy, ego.qz, ego.qw]).as_matrix()

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        #T[:3, 3] = -rot.T @ t  
        ego_cache[ego.token] = (aligned_ts, T)

    # 复制图像
    sensor_root = os.path.join(nuplan_root, 'nuplan-v1.1', 'sensor_blobs')
    print("[INFO] 导出 images（直接复制 + 重命名）...")
    for new_idx, img_obj, ts, cam_id, filename_jpg in tqdm(ts_slice, desc='copy'):

        ego_ts, ego_T = ego_cache.get(img_obj.ego_pose_token, (0., np.eye(4)))
        ego_file = os.path.join(ego_dir, f"{new_idx:06d}.txt")
        if not (skip_existing and os.path.exists(ego_file)):
            np.savetxt(ego_file, ego_T, fmt='%.8f')

        cam = db.camera.get(img_obj.camera_token) 
        cam_pose = ego_T @ cam.trans_matrix   # 官方 4×4 直接乘
        cam_file = os.path.join(ego_dir, f"{new_idx:06d}_{cam_id}.txt")
        if not (skip_existing and os.path.exists(cam_file)):
            np.savetxt(cam_file, cam_pose, fmt='%.8f')

        out_name = f"{new_idx:06d}_{cam_id}.png"
        out_path = os.path.join(img_dir, out_name)
        if skip_existing and os.path.exists(out_path):
            continue
        src_path = os.path.join(sensor_root, filename_jpg)
        if not os.path.isfile(src_path):
            print(f"[WARN] 缺失源图: {src_path}")
            continue
        # 读取相机内参
        cam = db.camera.get(img_obj.camera_token)
        K = cam.intrinsic_np
        D = cam.distortion_np
        # 去畸变并保持 1920×1080
        dst,new_K = undistort_image(src_path, K, D, dst_size=(1920, 1080))
        cam_newK_dict[cam_id] = new_K
        cv2.imwrite(out_path, dst)
    print("[DONE] 转换完成 ->", save_dir)

    print("[INFO] 导出相机标定 ...")
    for cam in db.camera:
        cam_id = cam_id_map.get(cam.channel, 0)

        K = cam.intrinsic_np          # 3×3
        D = cam.distortion_np         # 向量，长度 5 或 8
        waymo_vec = np.array([
            K[0,0], K[1,1], K[0,2], K[1,2],
            D[0], D[1], D[2], D[3], D[4] if len(D) >= 5 else 0.
        ]).reshape(9, 1)
        np.savetxt(os.path.join(intr_dir, f"{cam_id}.txt"), waymo_vec, fmt='%.8f')

        # 外参：4×4 直接拿官方矩阵
        E = cam.trans_matrix          # 官方 4×4
        np.savetxt(os.path.join(ext_dir, f"{cam_id}.txt"),
                E, fmt='%.8f')

    print("[INFO] 相机标定导出完成")




# ---------- 入口 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuplan_root', required=True, help='nuPlan dataset root (data_root)')
    parser.add_argument('--log_name', required=True, help='log name token')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    main(args.nuplan_root, args.log_name, args.save_dir, skip_existing=args.skip_existing)