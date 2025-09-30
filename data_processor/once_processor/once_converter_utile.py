#!/usr/bin/env python3
'''
import os
import json
import cv2
import numpy as np
import pickle
import argparse
import imageio
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
from concurrent.futures import ThreadPoolExecutor

# ---------- 参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="ONCE to StreetCraft (compatible with waymo_helpers)")
    parser.add_argument('--input_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx',   type=int, default=None)
    return parser.parse_args()

args = parse_args()
INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
CAMERA_NAMES = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
IMAGE_SIZE   = (1920, 1020)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 工具 ----------
def mkdir_p(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

mkdir_p(OUTPUT_DIR,
        f"{OUTPUT_DIR}/images",
        f"{OUTPUT_DIR}/ego_pose",
        f"{OUTPUT_DIR}/intrinsics",
        f"{OUTPUT_DIR}/extrinsics",
        f"{OUTPUT_DIR}/lidar",
        f"{OUTPUT_DIR}/track",
        f"{OUTPUT_DIR}/dynamic_mask")

with open(os.path.join(INPUT_DIR, [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')][0]), 'r') as f:
    DATA = json.load(f)
FRAMES = DATA['frames'][args.start_idx : args.end_idx]

# ---------- 1. 标定 ----------
def save_calib():
    for i, cam in enumerate(tqdm(CAMERA_NAMES, desc="calib")):
        cam_data = DATA['calib'][cam]
        intrinsic = np.array(cam_data['cam_intrinsic'])  # 3×3
        extrinsic = np.array(cam_data['cam_to_velo'])    # 4×4  cam_to_ego

        # extrinsics
        np.savetxt(f"{OUTPUT_DIR}/extrinsics/{i}.txt", extrinsic)

        # ✅ 保存原始 9 行内参
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        k1, k2, p1, p2, k3 = cam_data['distortion']
        vec = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3])
        np.savetxt(f"{OUTPUT_DIR}/intrinsics/{i}.txt", vec.reshape(-1, 1), fmt='%.18e')
# ---------- 2. 图像去畸变 ----------
def undistort_image(img_path, cam_idx, dst_path):
    intrinsic = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt")
    K = np.array([[intrinsic[0], 0, intrinsic[2]],
                  [0, intrinsic[1], intrinsic[3]],
                  [0, 0, 1]])
    dist = np.array(DATA['calib'][CAMERA_NAMES[cam_idx]]['distortion'])
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undist = cv2.undistort(img, K, dist, None, new_K)
    cv2.imwrite(dst_path, undist)

def save_images_undistort():
    tasks = []
    for idx, frame in enumerate(FRAMES):
        frame_id = frame['frame_id']
        for cam_idx, cam in enumerate(CAMERA_NAMES):
            src = f"{INPUT_DIR}/{cam}/{frame_id}.jpg"
            dst = f"{OUTPUT_DIR}/images/{idx:06d}_{cam_idx}.jpg"
            tasks.append((src, cam_idx, dst))
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(tqdm(ex.map(lambda x: undistort_image(*x), tasks), total=len(tasks), desc="undistort"))

# ---------- 3. 点云 ----------
def save_lidar():
    for idx, frame in enumerate(tqdm(FRAMES, desc="lidar")):
        frame_id = frame['frame_id']
        src = f"{INPUT_DIR}/lidar_roof/{frame_id}.bin"
        dst = f"{OUTPUT_DIR}/lidar/{idx:06d}.bin"
        if os.path.exists(src):
            pts = np.fromfile(src, dtype=np.float32).reshape(-1, 4)
            pts.tofile(dst)

# ---------- 4. 自车位姿 ----------
def save_poses():
    for idx, frame in enumerate(tqdm(FRAMES, desc="ego_pose")):
        q = np.array(frame['pose'][:4])
        t = np.array(frame['pose'][4:])
        rot = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        np.savetxt(f"{OUTPUT_DIR}/ego_pose/{idx:06d}.txt", T)

# ---------- 5. 轨迹 ----------
def build_trajectory():
    from scipy.optimize import linear_sum_assignment
    trajectory = {}
    track_info = {}
    track_cam_vis = {}
    tid_counter = 0
    active_tracks = {}
    MAX_MISS = 5

    def match(tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tid_list = list(tracks.keys())
        cost = np.zeros((len(tracks), len(dets)))
        for i, tid in enumerate(tid_list):
            box1 = tracks[tid]['box']
            for j, det in enumerate(dets):
                box2 = det['box']
                dist = np.linalg.norm(np.array(box1[:3]) - np.array(box2[:3]))
                size_diff = np.abs(np.array(box1[3:6]) - np.array(box2[3:6])).sum()
                dyaw = abs(box1[6] - box2[6])
                dyaw = min(dyaw, 2 * np.pi - dyaw)
                if dist > 3.0:
                    c = 1e5
                else:
                    c = dist + 0.3 * size_diff + 0.5 * dyaw
                cost[i, j] = c
        row, col = linear_sum_assignment(cost)
        matched, ut, ud = [], list(range(len(tracks))), list(range(len(dets)))
        for r, c in zip(row, col):
            if cost[r, c] < 2.5:
                matched.append((tid_list[r], c))
                ut.remove(r)
                ud.remove(c)
        return matched, ut, ud

    for idx, frame in enumerate(tqdm(FRAMES, desc="tracking")):
        fid = f"{idx:06d}"
        track_info[fid] = {}
        track_cam_vis[fid] = {i: [] for i in range(len(CAMERA_NAMES))}
        if 'annos' not in frame:
            for tid in list(active_tracks.keys()):
                active_tracks[tid]['miss_cnt'] += 1
            continue
        names, boxes = frame['annos']['names'], frame['annos']['boxes_3d']
        dets = [{'label': n.lower(), 'box': b} for n, b in zip(names, boxes)]
        matched, ut, ud = match({tid: {'box': tr['box']} for tid, tr in active_tracks.items()}, dets)
        for tid, d_idx in matched:
            det = dets[d_idx]
            active_tracks[tid]['box'] = det['box']
            active_tracks[tid]['frame_idx'] = idx
            active_tracks[tid]['miss_cnt'] = 0
        for d_idx in ud:
            det = dets[d_idx]
            tid = f"T{tid_counter:04d}"
            tid_counter += 1
            active_tracks[tid] = {'box': det['box'], 'frame_idx': idx, 'miss_cnt': 0}
        for ut_idx in ut:
            tid = list(active_tracks.keys())[ut_idx]
            active_tracks[tid]['miss_cnt'] += 1
        for tid in [k for k, v in active_tracks.items() if v['miss_cnt'] > MAX_MISS]:
            del active_tracks[tid]
        for tid, d_idx in matched:
            det = dets[d_idx]
            box = det['box']
            if tid not in trajectory:
                trajectory[tid] = {
                    'label': 'car',
                    'length': box[3],
                    'width': box[4],
                    'height': box[5],
                    'poses_vehicle': [],
                    'frames': [],
                    'timestamps': [],
                    'stationary': False,
                }
            pose = np.eye(4)
            yaw = box[6]
            rot = R.from_euler('z', yaw).as_matrix()
            pose[:3, :3] = rot
            pose[:3, 3] = box[:3]
            trajectory[tid]['poses_vehicle'].append(pose.astype(np.float32))
            trajectory[tid]['frames'].append(idx)
            trajectory[tid]['timestamps'].append(float(frame['frame_id']))
            track_cam_vis[fid][0].append(tid)  # 简化：全写 cam0
            track_info[fid][tid] = {
                'lidar_box': {
                    'center_x': box[0],
                    'center_y': box[1],
                    'center_z': box[2],
                    'length': box[3],
                    'width': box[4],
                    'height': box[5],
                    'heading': box[6],
                    'label': 'car',
                }
            }

    with open(f"{OUTPUT_DIR}/track/trajectory.pkl", 'wb') as f:
        pickle.dump(trajectory, f)
    with open(f"{OUTPUT_DIR}/track/track_info.pkl", 'wb') as f:
        pickle.dump(track_info, f)
    with open(f"{OUTPUT_DIR}/track/track_camera_visible.pkl", 'wb') as f:
        pickle.dump(track_cam_vis, f)
    with open(f"{OUTPUT_DIR}/track/track_ids.json", 'w') as f:
        json.dump({tid: tid for tid in trajectory}, f, indent=2)
    return trajectory, track_cam_vis

# ---------- 6. 动态掩码 ----------
def project_lidar_to_image(pts, T_velo2cam, K, img_size=(1920, 1020)):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int)

def save_dynamic_mask_gpu(trajectory, track_cam_vis):
    faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    for idx, frame in enumerate(tqdm(FRAMES, desc="dynamic_mask")):
        fid = f"{idx:06d}"
        for cam_i in range(len(CAMERA_NAMES)):
            mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
            intr = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_i}.txt")
            K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])
            T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{cam_i}.txt"))
            for tid in track_cam_vis[fid][0]:  # 简化：仅 cam0
                if tid not in trajectory or trajectory[tid]['stationary']:
                    continue
                if idx not in trajectory[tid]['frames']:
                    continue
                pose = trajectory[tid]['poses_vehicle'][trajectory[tid]['frames'].index(idx)]
                l, w, h = trajectory[tid]['length'], trajectory[tid]['width'], trajectory[tid]['height']
                corners_obj = 0.5 * np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
                                              [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])
                corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]
                uv = project_lidar_to_image(corners_v, T_velo2cam, K)
                if len(uv) < 8:
                    continue
                for f in faces:
                    cv2.fillPoly(mask, [uv[f, :].astype(np.int32)], color=255)
            cv2.imwrite(f"{OUTPUT_DIR}/dynamic_mask/{fid}_{cam_i}.jpg", mask)

# ---------- 7. 轨迹可视化（Waymo 风格） ----------
def save_track_vis(trajectory, track_cam_vis):
    import imageio
    # 默认用 ONCE 的 cam0（front），可改成 0~6
    CAM_VIS = 0
    intr = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{CAM_VIS}.txt")
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{CAM_VIS}.txt"))

    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    frames_vis = []
    for idx, frame in enumerate(tqdm(FRAMES, desc="track_vis")):
        img_path = f"{OUTPUT_DIR}/images/{idx:06d}_{CAM_VIS}.jpg"
        img = cv2.imread(img_path)
        if img is None:
            continue
        fid = f"{idx:06d}"

        for tid in track_cam_vis.get(fid, {}).get(CAM_VIS, []):
            if tid not in trajectory or idx not in trajectory[tid]['frames']:
                continue
            traj = trajectory[tid]
            pose_idx = traj['frames'].index(idx)
            pose = traj['poses_vehicle'][pose_idx]
            l, w, h = traj['length'], traj['width'], traj['height']

            # 8 角点（物体中心系 → 车体系 → 相机系）
            corners_obj = 0.5 * np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
                                          [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])
            corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]
            corners_c = (T_velo2cam @ np.hstack([corners_v, np.ones((8, 1))]).T).T[:, :3]
            # 投影
            uv = (K @ corners_c.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            # 只画在图像内的框
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & \
                     (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
            if inside.sum() < 8:
                continue
            uv = uv.astype(int)

            # 画 12 条边
            for i, j in edges:
                cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 0, 255), 2)
            # 画 Track ID（中心）
            ctr = uv.mean(axis=0).astype(int)
            cv2.putText(img, tid, tuple(ctr), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        frames_vis.append(img[:, :, ::-1])  # BGR → RGB

    out_path = f"{OUTPUT_DIR}/track/track_vis.mp4"
    imageio.mimwrite(out_path, frames_vis, fps=5)
    print(f"✅ 可视化视频已保存：{out_path}")

# ---------- main ----------
def main():
    if not FRAMES:
        print("⚠️  指定帧区间无数据，请检查 --start_idx / --end_idx")
        return
    save_calib()
    save_images_undistort()
    save_lidar()
    save_poses()
    traj, tcv = build_trajectory()
    save_dynamic_mask_gpu(traj, tcv)
    save_track_vis(traj, tcv)
    print("✅ ONCE → StreetCraft (waymo_helpers compatible) 完成！")


if __name__ == '__main__':
    main()
'''
#!/usr/bin/env python3
"""
ONCE → StreetCraft 格式（兼容 waymo_helpers.py）
跟踪：3D 恒加速度卡尔曼滤波（CA 模型）
"""
import os
import json
import cv2
import numpy as np
import pickle
import argparse
import imageio
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
from concurrent.futures import ThreadPoolExecutor

# ---------- 参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="ONCE to StreetCraft (CA-Kalman)")
    parser.add_argument('--input_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx',   type=int, default=None)
    return parser.parse_args()

args = parse_args()
INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
CAMERA_NAMES = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
IMAGE_SIZE   = (1920, 1020)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 工具 ----------
def mkdir_p(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

mkdir_p(OUTPUT_DIR,
        f"{OUTPUT_DIR}/images",
        f"{OUTPUT_DIR}/ego_pose",
        f"{OUTPUT_DIR}/intrinsics",
        f"{OUTPUT_DIR}/extrinsics",
        f"{OUTPUT_DIR}/lidar",
        f"{OUTPUT_DIR}/track",
        f"{OUTPUT_DIR}/dynamic_mask")

with open(os.path.join(INPUT_DIR, [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')][0]), 'r') as f:
    DATA = json.load(f)
FRAMES = DATA['frames'][args.start_idx : args.end_idx]

# ======================== 恒加速度卡尔曼滤波器 ========================
class KalmanBox3D_CA:
    def __init__(self, box):
        # 状态：x,y,z,l,w,h,yaw,vx,vy,vz,ax,ay,az
        self.x = np.zeros(13, dtype=np.float32)
        self.x[:7] = box
        self.P = np.eye(13, dtype=np.float32)
        self.Q = np.eye(13, dtype=np.float32) * 0.01
        self.Q[7:10, 7:10] *= 5
        self.Q[10:, 10:] *= 10
        self.R = np.eye(7, dtype=np.float32) * 0.1
        self.F = np.eye(13, dtype=np.float32)

    def predict(self):
        dt = 1.0
        self.F[0, 7] = dt
        self.F[1, 8] = dt
        self.F[2, 9] = dt
        self.F[0, 10] = 0.5 * dt**2
        self.F[1, 11] = 0.5 * dt**2
        self.F[2, 12] = 0.5 * dt**2
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        H = np.zeros((7, 13), dtype=np.float32)
        H[:7, :7] = np.eye(7)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(13) - K @ H) @ self.P

    @property
    def box(self):
        return self.x[:7].copy()
# ====================================================================

# ---------- 1. 标定 ----------
def save_calib():
    for i, cam in enumerate(tqdm(CAMERA_NAMES, desc="calib")):
        cam_data = DATA['calib'][cam]
        intrinsic = np.array(cam_data['cam_intrinsic'])
        extrinsic = np.array(cam_data['cam_to_velo'])
        np.savetxt(f"{OUTPUT_DIR}/extrinsics/{i}.txt", extrinsic)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        k1, k2, p1, p2, k3 = cam_data['distortion']
        vec = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3])
        np.savetxt(f"{OUTPUT_DIR}/intrinsics/{i}.txt", vec.reshape(-1, 1), fmt='%.18e')

# ---------- 2. 图像去畸变 ----------
def undistort_image(img_path, cam_idx, dst_path):
    intrinsic = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt")
    K = np.array([[intrinsic[0], 0, intrinsic[2]],
                  [0, intrinsic[1], intrinsic[3]],
                  [0, 0, 1]])
    dist = np.array(DATA['calib'][CAMERA_NAMES[cam_idx]]['distortion'])
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undist = cv2.undistort(img, K, dist, None, new_K)
    cv2.imwrite(dst_path, undist)

def save_images_undistort():
    tasks = []
    for idx, frame in enumerate(FRAMES):
        frame_id = frame['frame_id']
        for cam_idx, cam in enumerate(CAMERA_NAMES):
            src = f"{INPUT_DIR}/{cam}/{frame_id}.jpg"
            dst = f"{OUTPUT_DIR}/images/{idx:06d}_{cam_idx}.jpg"
            tasks.append((src, cam_idx, dst))
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(tqdm(ex.map(lambda x: undistort_image(*x), tasks), total=len(tasks), desc="undistort"))

# ---------- 3. 点云 ----------
def save_lidar():
    for idx, frame in enumerate(tqdm(FRAMES, desc="lidar")):
        frame_id = frame['frame_id']
        src = f"{INPUT_DIR}/lidar_roof/{frame_id}.bin"
        dst = f"{OUTPUT_DIR}/lidar/{idx:06d}.bin"
        if os.path.exists(src):
            pts = np.fromfile(src, dtype=np.float32).reshape(-1, 4)
            pts.tofile(dst)

# ---------- 4. 自车位姿 ----------
def save_poses():
    for idx, frame in enumerate(tqdm(FRAMES, desc="ego_pose")):
        q = np.array(frame['pose'][:4])
        t = np.array(frame['pose'][4:])
        rot = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        np.savetxt(f"{OUTPUT_DIR}/ego_pose/{idx:06d}.txt", T)

# ---------- 5. 轨迹（CA 卡尔曼） ----------
def build_trajectory():
    from scipy.optimize import linear_sum_assignment
    trajectory = {}
    track_info = {}
    track_cam_vis = {}
    tid_counter = 0
    active_tracks = {}
    MAX_MISS = 5

    def match(tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tid_list = list(tracks.keys())
        cost = np.zeros((len(tracks), len(dets)))
        for i, tid in enumerate(tid_list):
            box1 = tracks[tid]['kf'].box
            for j, det in enumerate(dets):
                box2 = det['box']
                dist = np.linalg.norm(np.array(box1[:3]) - np.array(box2[:3]))
                size_diff = np.abs(np.array(box1[3:6]) - np.array(box2[3:6])).sum()
                dyaw = abs(box1[6] - box2[6])
                dyaw = min(dyaw, 2 * np.pi - dyaw)
                if dist > 3.0:
                    c = 1e5
                else:
                    c = dist + 0.3 * size_diff + 0.5 * dyaw
                cost[i, j] = c
        row, col = linear_sum_assignment(cost)
        matched, ut, ud = [], list(range(len(tracks))), list(range(len(dets)))
        for r, c in zip(row, col):
            if cost[r, c] < 2.5:
                matched.append((tid_list[r], c))
                ut.remove(r)
                ud.remove(c)
        return matched, ut, ud

    for idx, frame in enumerate(tqdm(FRAMES, desc="tracking")):
        fid = f"{idx:06d}"
        track_info[fid] = {}
        track_cam_vis[fid] = {i: [] for i in range(len(CAMERA_NAMES))}
        if 'annos' not in frame:
            for tid in list(active_tracks.keys()):
                active_tracks[tid]['miss_cnt'] += 1
            continue
        names, boxes = frame['annos']['names'], frame['annos']['boxes_3d']
        dets = [{'label': n.lower(), 'box': b} for n, b in zip(names, boxes)]
        # 先预测
        for tid in active_tracks:
            active_tracks[tid]['kf'].predict()
        matched, ut, ud = match(active_tracks, dets)
        for tid, d_idx in matched:
            det = dets[d_idx]
            active_tracks[tid]['kf'].update(det['box'])
            active_tracks[tid]['frame_idx'] = idx
            active_tracks[tid]['miss_cnt'] = 0
        for d_idx in ud:
            det = dets[d_idx]
            tid = f"T{tid_counter:04d}"
            tid_counter += 1
            kf = KalmanBox3D_CA(det['box'])
            active_tracks[tid] = {'kf': kf, 'frame_idx': idx, 'miss_cnt': 0}
        for ut_idx in ut:
            tid = list(active_tracks.keys())[ut_idx]
            active_tracks[tid]['miss_cnt'] += 1
        for tid in [k for k, v in active_tracks.items() if v['miss_cnt'] > MAX_MISS]:
            del active_tracks[tid]
        # 保存轨迹
        for tid, d_idx in matched:
            det = dets[d_idx]
            box = active_tracks[tid]['kf'].box
            if tid not in trajectory:
                trajectory[tid] = {
                    'label': 'car',
                    'length': box[3],
                    'width': box[4],
                    'height': box[5],
                    'poses_vehicle': [],
                    'frames': [],
                    'timestamps': [],
                    'stationary': False,
                }
            pose = np.eye(4)
            yaw = box[6]
            rot = R.from_euler('z', yaw).as_matrix()
            pose[:3, :3] = rot
            pose[:3, 3] = box[:3]
            trajectory[tid]['poses_vehicle'].append(pose.astype(np.float32))
            trajectory[tid]['frames'].append(idx)
            trajectory[tid]['timestamps'].append(float(frame['frame_id']))
            track_cam_vis[fid][0].append(tid)  # 简化 cam0
            track_info[fid][tid] = {
                'lidar_box': {
                    'center_x': box[0],
                    'center_y': box[1],
                    'center_z': box[2],
                    'length': box[3],
                    'width': box[4],
                    'height': box[5],
                    'heading': box[6],
                    'label': 'car',
                }
            }

    with open(f"{OUTPUT_DIR}/track/trajectory.pkl", 'wb') as f:
        pickle.dump(trajectory, f)
    with open(f"{OUTPUT_DIR}/track/track_info.pkl", 'wb') as f:
        pickle.dump(track_info, f)
    with open(f"{OUTPUT_DIR}/track/track_camera_visible.pkl", 'wb') as f:
        pickle.dump(track_cam_vis, f)
    with open(f"{OUTPUT_DIR}/track/track_ids.json", 'w') as f:
        json.dump({tid: tid for tid in trajectory}, f, indent=2)
    return trajectory, track_cam_vis

# ---------- 6. 动态掩码 ----------
def project_lidar_to_image(pts, T_velo2cam, K, img_size=(1920, 1020)):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int)

def save_dynamic_mask_gpu(trajectory, track_cam_vis):
    faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    for idx, frame in enumerate(tqdm(FRAMES, desc="dynamic_mask")):
        fid = f"{idx:06d}"
        for cam_i in range(len(CAMERA_NAMES)):
            mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)
            intr = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_i}.txt")
            K = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]])
            T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{cam_i}.txt"))
            for tid in track_cam_vis[fid][0]:
                if tid not in trajectory or trajectory[tid]['stationary']:
                    continue
                if idx not in trajectory[tid]['frames']:
                    continue
                pose = trajectory[tid]['poses_vehicle'][trajectory[tid]['frames'].index(idx)]
                l, w, h = trajectory[tid]['length'], trajectory[tid]['width'], trajectory[tid]['height']
                corners_obj = 0.5 * np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
                                              [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])
                corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]
                uv = project_lidar_to_image(corners_v, T_velo2cam, K)
                if len(uv) < 8:
                    continue
                for f in faces:
                    cv2.fillPoly(mask, [uv[f, :].astype(np.int32)], color=255)
            cv2.imwrite(f"{OUTPUT_DIR}/dynamic_mask/{fid}_{cam_i}.jpg", mask)

# ---------- 7. 轨迹可视化 ----------
def save_track_vis(trajectory, track_cam_vis):
    CAM_VIS = 0
    intr = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{CAM_VIS}.txt")
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{CAM_VIS}.txt"))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    frames_vis = []
    for idx, frame in enumerate(tqdm(FRAMES, desc="track_vis")):
        img_path = f"{OUTPUT_DIR}/images/{idx:06d}_{CAM_VIS}.jpg"
        img = cv2.imread(img_path)
        if img is None:
            continue
        fid = f"{idx:06d}"
        for tid in track_cam_vis.get(fid, {}).get(CAM_VIS, []):
            if tid not in trajectory or idx not in trajectory[tid]['frames']:
                continue
            traj = trajectory[tid]
            pose_idx = traj['frames'].index(idx)
            pose = traj['poses_vehicle'][pose_idx]
            l, w, h = traj['length'], traj['width'], traj['height']
            corners_obj = 0.5 * np.array([[-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
                                          [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]])
            corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]
            corners_c = (T_velo2cam @ np.hstack([corners_v, np.ones((8, 1))]).T).T[:, :3]
            uv = (K @ corners_c.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & \
                     (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
            if inside.sum() < 8:
                continue
            uv = uv.astype(int)
            for i, j in edges:
                cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 0, 255), 2)
            ctr = uv.mean(axis=0).astype(int)
            cv2.putText(img, tid, tuple(ctr), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        frames_vis.append(img[:, :, ::-1])
    out_path = f"{OUTPUT_DIR}/track/track_vis.mp4"
    imageio.mimwrite(out_path, frames_vis, fps=5)
    print(f"✅ 可视化视频已保存：{out_path}")

# ---------- main ----------
def main():
    if not FRAMES:
        print("⚠️  指定帧区间无数据，请检查 --start_idx / --end_idx")
        return
    save_calib()
    save_images_undistort()
    save_lidar()
    save_poses()
    traj, tcv = build_trajectory()
    save_dynamic_mask_gpu(traj, tcv)
    save_track_vis(traj, tcv)
    print("✅ ONCE → StreetCraft（CA-Kalman）完成！")

if __name__ == '__main__':
    main()