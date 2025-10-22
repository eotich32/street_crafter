#!/usr/bin/env python3
"""
nuplan-devkit 导出脚本
- 每相机 200 帧，时间戳不再四舍五入，采用“相邻最近”匹配
- 图像-点云、图像-图像全部按绝对最近原则配对
"""
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
import cv2, imageio
from copy import deepcopy
import pickle
import torch
import pandas as pd

from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB

# ---------- 工具 ----------
def makedirs(p):
    os.makedirs(p, exist_ok=True)

opencv2camera = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])

# 不再四舍五入，保持原始微秒
def identity_ts(ts_us):
    return int(ts_us)

# ---------- 去畸变 ----------
def undistort_image(img_path, K, D, dst_size=(1920, 1080)):
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((dst_size[1], dst_size[0], 3), np.uint8), K
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0, newImgSize=dst_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, dst_size, cv2.CV_16SC2)
    dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return dst, new_K

# ---------- 3D 投影 ----------
@torch.no_grad()
def project_visible_boxes(poses_v, dims, K, E, w, h, tid_list):
    device = K.device
    N = len(poses_v)
    if N == 0:
        return []
    T_ego2cam = torch.linalg.inv(E)
    x = torch.tensor([-1,1,1,-1,-1,1,1,-1], device=device)*0.5
    y = torch.tensor([1,1,-1,-1,1,1,-1,-1], device=device)*0.5
    z = torch.tensor([-1,-1,-1,-1,1,1,1,1], device=device)*0.5
    corners_template = torch.stack([x,y,z,torch.ones(8,device=device)])
    out = []
    for i in range(N):
        l, w, h = dims[i]
        S = torch.diag(torch.tensor([l, w, h, 1.], device=device))
        T = torch.from_numpy(poses_v[i]).to(device)
        corners = T @ S @ corners_template
        corners_cam = T_ego2cam @ corners
        pts2d = K @ corners_cam[:3]
        uv = pts2d[:2] / (pts2d[2:3] + 1e-6)
        valid = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h) & (corners_cam[2] > 0)
        if valid.any():
            out.append((uv.T.cpu().numpy(), valid.cpu().numpy(), tid_list[i]))
    return out

def compute_3d_box_corners(dim, pose):
    l, w, h = dim
    x = np.array([-l, l, l, -l, -l, l, l, -l])/2
    y = np.array([w, w, -w, -w, w, w, -w, -w]) / 2
    z = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])
    pts = np.stack([x, y, z, np.ones(8)], axis=0)
    pts = (pose @ pts)[:3, :].T
    return pts

# ---------- 相机 ID 映射 ----------
def _build_cam_id_map(db):
    cameras = getattr(db, 'camera', None) or (db.get('camera') if hasattr(db, 'get') else None) or []
    return {cam.channel: idx for idx, cam in enumerate(cameras)}

# ---------- 每相机 200 帧（原始时间戳） ----------
def _build_per_camera_frames(db, frames_per_cam=200, start_idx=0):
    Image = db.image[0].__class__
    query = (db.session.query(Image)
                       .order_by(Image.timestamp)
                       .offset(start_idx)
                       .limit(8 * frames_per_cam * 2))
    all_rows = query.all()
    cam_chunks = defaultdict(list)
    for img in all_rows:
        parts = img.filename_jpg.split('/')
        if len(parts) < 3:
            continue
        cam_name = parts[1]
        if len(cam_chunks[cam_name]) < frames_per_cam:
            cam_chunks[cam_name].append((img.timestamp, img))   # 原始 ts
    result = []
    cam_id_map = {name: idx for idx, name in enumerate(cam_chunks.keys())}
    for cam_name, ts_imgs in cam_chunks.items():
        cam_id = cam_id_map[cam_name]
        for new_idx, (ts, img) in enumerate(ts_imgs):
            result.append((new_idx, img, ts, cam_id, img.filename_jpg))
    return result, cam_id_map

# ---------- 点云最近查找表 ----------
def build_lidar_nearest_lut(pc_df):
    pc_ts = pc_df['timestamp'].values.astype(np.int64)
    pc_tok = pc_df['token'].values
    def find_nearest(ts_us):
        diff = np.abs(pc_ts - ts_us)
        idx = diff.argmin()
        return pc_tok[idx], pc_ts[idx]
    return find_nearest

# ---------- 内存表 ----------
def build_df(db):
    cam_df = pd.DataFrame([
        dict(channel=c.channel, K=c.intrinsic_np, D=c.distortion_np,
             E=c.trans_matrix, width=c.width, height=c.height, token=c.token)
        for c in db.camera
    ])
    ego_df = pd.DataFrame([
        dict(token=e.token, timestamp=e.timestamp,
             x=e.x, y=e.y, z=e.z, qx=e.qx, qy=e.qy, qz=e.qz, qw=e.qw)
        for e in db.ego_pose
    ])
    pc_df = pd.DataFrame([
        dict(token=pc.token, ego_pose_token=pc.ego_pose_token, timestamp=pc.timestamp)
        for pc in db.lidar_pc
    ])
    track_df = pd.DataFrame([
        dict(token=t.token, category_token=t.category_token)
        for t in db.track
    ])
    box_df = pd.DataFrame([
        dict(token=b.token, lidar_pc_token=b.lidar_pc_token, track_token=b.track_token,
             x=b.x, y=b.y, z=b.z, width=b.width, length=b.length, height=b.height,
             yaw=b.yaw, vx=b.vx, vy=b.vy, vz=b.vz, confidence=b.confidence)
        for b in db.lidar_box
    ])
    box_df = box_df.merge(pc_df[['token','timestamp']].rename(columns={'token':'lidar_pc_token'}),
                          on='lidar_pc_token', how='left')
    box_df = box_df.merge(track_df[['token','category_token']].rename(columns={'token':'track_token'}),
                          on='track_token', how='left')
    cat_df = pd.DataFrame([dict(token=c.token, name=c.name) for c in db.category])
    box_df = box_df[box_df.category_token.map(
        cat_df.set_index('token').name.isin(['vehicle','bicycle','pedestrian'])
    )]
    return cam_df, ego_df, pc_df, box_df, cat_df

# ---------- GPU 批量投影 ----------
@torch.no_grad()
def batch_project_to_cam(boxes_ego, Ks, Es, widths, heights):
    N = boxes_ego.shape[0]
    C = Ks.shape[0]
    pts = torch.cat([boxes_ego, torch.ones(N,1, device=boxes_ego.device)], dim=1)
    pts_cam = Es @ pts.T
    pts_cam = pts_cam[:,:3]
    z = pts_cam[:,2]
    uv = Ks @ pts_cam
    uv = uv[:,:2]/(uv[:,2:3]+1e-6)
    u,v = uv[:,0], uv[:,1]
    visible = (z>0)&(u>=0)&(u<widths.view(-1,1))&(v>=0)&(v<heights.view(-1,1))
    return visible.T

# ---------- 主导出（GPU） ----------
def export_track_trajectory_gpu(db, ts_slice, cam_id_map, ego_cache, save_dir,
                                skip_existing=False, cam_newK_dict=None,valid_cam_ids=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    track_dir = os.path.join(save_dir, 'track'); os.makedirs(track_dir, exist_ok=True)
    if cam_newK_dict is None:
        print("去畸变后的相机外参没有正确传入函数")

    cam_df, ego_df, pc_df, box_df, cat_df = build_df(db)
    ep2newidx = {img.ego_pose_token: new_idx for new_idx, img, _, _, _ in ts_slice}
    pc2ep = dict(zip(pc_df.token, pc_df.ego_pose_token))
    box_df = box_df[box_df.lidar_pc_token.map(pc2ep).isin(ep2newidx)]
    box_df['ep'] = box_df.lidar_pc_token.map(pc2ep)
    box_df['new_idx'] = box_df.ep.map(ep2newidx)

    cam_df['cam_id'] = cam_df.channel.map(cam_id_map)
    cam_df = cam_df.sort_values('cam_id')
    Ks = torch.stack([torch.from_numpy(np.array(c.K, dtype=np.float32)) for _, c in cam_df.iterrows()]).to(device)
    Es = torch.stack([torch.from_numpy(np.array(c.E, dtype=np.float32)) for _, c in cam_df.iterrows()]).to(device)
    widths = torch.tensor(cam_df.width.values, device=device)
    heights = torch.tensor(cam_df.height.values, device=device)

    world2ego_cache = {ego.token: np.linalg.inv(ego_cache[ego.token][1]) for ego in db.ego_pose}

    CONF_THR = 0.4
    V_STILL = 0.5
    box_df['speed'] = np.hypot(box_df.vx, box_df.vy)
    box_df['stationary'] = box_df.speed <= V_STILL
    box_df = box_df[box_df.confidence >= CONF_THR]

    frames = defaultdict(list)
    for _, row in box_df.iterrows():
        frames[row.new_idx].append(row)

    track_info = defaultdict(dict)
    track_camera_visible = defaultdict(lambda: defaultdict(list))
    trajectory = {}

    cam_param_cpu = {}
    for _, row in cam_df.iterrows():
        cam_id = int(row.cam_id)
        K_use = cam_newK_dict.get(cam_id, np.array(row.K, dtype=np.float64))
        cam_param_cpu[cam_id] = dict(K=K_use, D=np.zeros(5),
                                     E=np.array(row.E, dtype=np.float64),
                                     w=int(row.width), h=int(row.height))

    for new_idx, rows in tqdm(frames.items(), desc='gpu track'):
        frame_key = f"{new_idx:06d}"
        for r in rows:
            ego_T = world2ego_cache[r.ep]
            center_world = np.array([r.x, r.y, r.z, 1.0])
            R_world_to_ego = ego_T[:3, :3]
            yaw_world = r.yaw
            R_obj_world = Rotation.from_euler('z', yaw_world).as_matrix()
            R_obj_ego = R_world_to_ego @ R_obj_world
            yaw_ego = Rotation.from_matrix(R_obj_ego).as_euler('xyz')[2]
            center_ego = (ego_T @ center_world)[:3]
            box_dict = dict(center_x=float(center_ego[0]),
                            center_y=float(center_ego[1]),
                            center_z=float(center_ego[2]),
                            width=r.width, length=r.length, height=r.height,
                            #heading=r.yaw,
                            heading=yaw_ego,
                            velocity=np.array([r.vx, r.vy, r.vz]),
                            label=cat_df.set_index('token').loc[r.category_token, 'name'],
                            timestamp=ego_cache[r.ep][0])
            track_info[frame_key][r.track_token] = box_dict

            if r.track_token not in trajectory:
                trajectory[r.track_token] = dict(label=box_dict['label'],
                                                 frames=[], poses_vehicle=[], poses_world=[],
                                                 timestamps=[], height=r.height, width=r.width,
                                                 length=r.length, stationary=True, deformable=False,
                                                 symmetric=True)
            traj = trajectory[r.track_token]
            T_obj_ego = np.array(
                [[np.cos(yaw_ego), -np.sin(yaw_ego), 0, center_ego[0]],
                 [np.sin(yaw_ego), np.cos(yaw_ego), 0, center_ego[1]],
                 [0, 0, 1, center_ego[2]],
                 [0, 0, 0, 1]], dtype=np.float32)
            corners_ego = compute_3d_box_corners([r.length, r.width, r.height], T_obj_ego)
            corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
            for cam_id, cam_p in cam_param_cpu.items():
                K, D, E, w, h = cam_p['K'], cam_p['D'], cam_p['E'], cam_p['w'], cam_p['h']
                ego2cam = np.linalg.inv(E)
                pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), K, D)
                uv = uv.squeeze()
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (pts_cam[:, 2] > 0)
                if inside.any():
                    track_camera_visible[frame_key][cam_id].append(r.track_token)

    for t_token, sub in box_df.groupby('track_token'):
        sub = sub.sort_values('timestamp')
        frames = sub.new_idx.tolist()
        poses_v, poses_w = [], []
        for _, r in sub.iterrows():
            ego_T = world2ego_cache[r.ep]
            R_world_to_ego = ego_T[:3, :3]
            yaw_world = r.yaw
            R_obj_world = Rotation.from_euler('z', yaw_world).as_matrix()
            R_obj_ego = R_world_to_ego @ R_obj_world
            yaw_ego = Rotation.from_matrix(R_obj_ego).as_euler('xyz')[2]
            center_world = np.array([r.x, r.y, r.z, 1.0])
            center_ego = (ego_T @ center_world)[:3]
            T_obj_ego = np.array(
                [[np.cos(yaw_ego), -np.sin(yaw_ego), 0, center_ego[0]],
                 [np.sin(yaw_ego), np.cos(yaw_ego), 0, center_ego[1]],
                 [0, 0, 1, center_ego[2]],
                 [0, 0, 0, 1]], dtype=np.float32)
            T_obj_world = ego_T @ T_obj_ego
            poses_v.append(T_obj_ego.astype(np.float32))
            poses_w.append(T_obj_world.astype(np.float32))
        dim = np.array([sub.height.max(), sub.width.max(), sub.length.max()], dtype=np.float32)
        label = cat_df.set_index('token').loc[sub.category_token.iloc[0], 'name']
        stationary = bool(box_df[box_df.track_token == t_token].stationary.all())
        if len(frames) < 2:
            continue
        trajectory[t_token] = dict(
            label=label,
            height=dim[0], width=dim[1], length=dim[2],
            poses_vehicle=np.array(poses_v),
            frames=frames,
            timestamps=sub.timestamp.tolist(),
            stationary=stationary,
            symmetric=(label != 'pedestrian'),
            deformable=(label == 'pedestrian')
        )

    with open(os.path.join(track_dir, 'track_info.pkl'), 'wb') as f:
        pickle.dump(dict(track_info), f)
    with open(os.path.join(track_dir, 'track_camera_visible.pkl'), 'wb') as f:
        pickle.dump(dict(track_camera_visible), f)
    with open(os.path.join(track_dir, 'trajectory.pkl'), 'wb') as f:
        pickle.dump(trajectory, f)
    with open(os.path.join(track_dir, 'track_ids.json'), 'w') as f:
        json.dump({t: i for i, t in enumerate(trajectory.keys())}, f, indent=2)
    print('[INFO] GPU track 导出完成')

    # ---------- 可视化 + dynamic_mask ----------
    print('[INFO] 开始画框 + 生成 dynamic_mask ...')
    vis_video_dir = os.path.join(save_dir, 'vis_videos'); os.makedirs(vis_video_dir, exist_ok=True)
    dynamic_mask_dir = os.path.join(save_dir, 'dynamic_mask'); os.makedirs(dynamic_mask_dir, exist_ok=True)
    CAM_NAMES = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0', 'CAM_L2', 'CAM_L1']
    #cam_id_list = [cam_id_map[name] for name in CAM_NAMES]
    cam_id_list = [cam_id_map[name] for name in CAM_NAMES
               if cam_id_map.get(name) in valid_cam_ids]
    if not cam_id_list:
        print('[WARN] 指定相机编号无有效通道，跳过可视化')
    else:
        fps, frame_size = 10, (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers = {cam_id: cv2.VideoWriter(
            os.path.join(vis_video_dir, f'{CAM_NAMES[i]}.mp4'),
            fourcc, fps, frame_size)
            for i, cam_id in enumerate(cam_id_list)}
        frame_to_img_path = {
            (new_idx, cam_id): os.path.join(save_dir, "images", f'{new_idx:06d}_{cam_id}.png')
            for new_idx, _, _, cam_id, _ in ts_slice
        }
        mask_cache = {}
        for new_idx, img_obj, _, _, _ in tqdm(ts_slice, desc='draw+mask'):
            frame_key = f'{new_idx:06d}'
            for cam_id in cam_id_list:
                img_path = frame_to_img_path.get((new_idx, cam_id))
                if not img_path or not os.path.exists(img_path):
                    continue
                canvas = cv2.imread(img_path)
                h, w = canvas.shape[:2]
                if cam_id not in mask_cache:
                    mask_cache[cam_id] = np.zeros((h, w), dtype=np.uint8)
                mask_canvas = mask_cache[cam_id]
                for tid in track_camera_visible.get(frame_key, {}).get(cam_id, []):
                    if tid not in trajectory:
                        continue
                    traj = trajectory[tid]
                    is_dynamic = not traj['stationary']
                    if new_idx not in traj['frames']:
                        continue
                    idx = traj['frames'].index(new_idx)
                    pose_v = traj['poses_vehicle'][idx]
                    dim = [traj['length'], traj['width'], traj['height']]
                    corners_ego = compute_3d_box_corners(dim, pose_v)
                    corners_ego_h = np.hstack([corners_ego, np.ones((8, 1))])
                    cam_p = cam_param_cpu[cam_id]
                    K, D, E = cam_p['K'], cam_p['D'], cam_p['E']
                    ego2cam = np.linalg.inv(E)
                    pts_cam = (ego2cam @ corners_ego_h.T)[:3].T
                    uv, _ = cv2.projectPoints(pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), K, D)
                    uv = uv.squeeze().astype(int)
                    valid = (pts_cam[:, 2] > 0)
                    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)]
                    for i, j in edges:
                        if valid[i] and valid[j]:
                            cv2.line(canvas, tuple(uv[i]), tuple(uv[j]), (0, 255, 0), 2)
                    if is_dynamic and valid.sum() >= 3:
                        hull = cv2.convexHull(uv[valid])
                        cv2.fillPoly(mask_canvas, [hull], 255)
                writers[cam_id].write(canvas)
                mask_path = os.path.join(dynamic_mask_dir, f'{new_idx:06d}_{cam_id}.png')
                cv2.imwrite(mask_path, mask_canvas)
                mask_cache[cam_id][:] = 0
        for w in writers.values():
            w.release()
        print('[INFO] 视频已保存 ->', vis_video_dir)
        print('[INFO] dynamic_mask 已保存 ->', dynamic_mask_dir)

# ---------- 主流程 ----------
def main(nuplan_root, log_name, save_dir, skip_existing=False,frame_nums=None, cam_ids=None):
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


    ts_slice, cam_id_map = _build_per_camera_frames(db, frames_per_cam=frame_nums, start_idx=0)
    print("[INFO] 相机通道映射:", cam_id_map)

    # 放在解析完 db 之后，拿到 cam_id_map 即可
    valid_cam_ids = set(cam_ids) if cam_ids is not None else set(cam_id_map.values())
    print(valid_cam_ids)
    if cam_ids is not None and not valid_cam_ids.issubset(cam_id_map.values()):
        raise ValueError(f'无效相机编号 {cam_ids}，合法编号 {list(cam_id_map.values())}')

    # timestamps.json（原始时间戳）
    ts_nested = defaultdict(dict)
    for idx, _, ts, cam_id, _ in ts_slice:
        if cam_id not in valid_cam_ids:
            continue
        cam_name = [k for k, v in cam_id_map.items() if v == cam_id][0]
        ts_nested[cam_name][f"{idx:06d}"] = float(ts)
    ts_json_path = os.path.join(save_dir, 'timestamps.json')
    with open(ts_json_path, 'w') as f:
        json.dump(ts_nested, f, indent=2)
    print(f"[INFO] timestamps.json 已写入，共 {len(ts_slice)} 条")

    ego_cache = {}
    for ego in db.ego_pose:
        raw_ts = identity_ts(ego.timestamp)
        t = np.array([ego.x, ego.y, ego.z])
        rot = Rotation.from_quat([ego.qx, ego.qy, ego.qz, ego.qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        ego_cache[ego.token] = (raw_ts, T)
    sensor_root = os.path.join(nuplan_root, 'nuplan-v1.1', 'sensor_blobs')
    print("[INFO] 导出 images（去畸变 + 重命名）...")
    
    for new_idx, img_obj, ts, cam_id, filename_jpg in tqdm(ts_slice, desc='copy'):
        if cam_id not in valid_cam_ids:      # ← 只处理指定相机
            continue
        ego_ts, ego_T = ego_cache.get(img_obj.ego_pose_token, (0., np.eye(4)))
        ego_file = os.path.join(ego_dir, f"{new_idx:06d}.txt")
        #if not (skip_existing and os.path.exists(ego_file)):
        if cam_id==0:
            np.savetxt(ego_file, ego_T, fmt='%.8f')

        cam = db.camera.get(img_obj.camera_token)
        cam_pose = ego_T #@ cam.trans_matrix
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
        K = cam.intrinsic_np
        D = cam.distortion_np
        dst, new_K = undistort_image(src_path, K, D, dst_size=(1920, 1080))
        cam_newK_dict[cam_id] = new_K
        cv2.imwrite(out_path, dst)
    print("[DONE] 图像转换完成")
    
    
    print("[INFO] 导出相机标定 ...")
    for cam in db.camera:
        cam_id = cam_id_map.get(cam.channel)
        if cam_id not in valid_cam_ids:      # ← 只保存指定相机
            continue
        cam_id = cam_id_map.get(cam.channel, 0)
        K = cam.intrinsic_np
        D = cam.distortion_np
        waymo_vec = np.array([
            K[0,0], K[1,1], K[0,2], K[1,2],
            D[0], D[1], D[2], D[3], D[4] if len(D) >= 5 else 0.
        ]).reshape(9, 1)
        np.savetxt(os.path.join(intr_dir, f"{cam_id}.txt"), waymo_vec, fmt='%.8f')
        E = cam.trans_matrix
        #E = np.matmul(E, opencv2camera)
        np.savetxt(os.path.join(ext_dir, f"{cam_id}.txt"), E, fmt='%.8f')
    print("[INFO] 相机标定导出完成")

    export_track_trajectory_gpu(db, ts_slice, cam_id_map,
                                ego_cache, save_dir,
                                skip_existing=skip_existing,
                                cam_newK_dict=cam_newK_dict,valid_cam_ids=valid_cam_ids)

# ---------- 入口 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuplan_root', required=True, help='nuPlan dataset root (data_root)')
    parser.add_argument('--log_name', required=True, help='log name token')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--frame_num', type=int,default=100, help='每相机帧数')
    parser.add_argument('--cam_ids', type=int, nargs='+', default=None,
                        help='仅导出指定相机编号（如 0 1 3），不指定则导出全部')

    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir,args.log_name)
    main(args.nuplan_root, args.log_name, save_dir,
         skip_existing=args.skip_existing, frame_nums=args.frame_num,cam_ids=args.cam_ids)