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
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve


# ---------- 参数 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="ONCE to StreetCraft (GPU acc)")
    parser.add_argument('--input_dir',  type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0,
                        help="起始帧序号（含）")
    parser.add_argument('--end_idx',   type=int, default=None,
                        help="结束帧序号（不含）；留空则到结尾")
    parser.add_argument('--cams', type=str, default=None,
                        help='要处理的相机，英文逗号分隔，例如 '
                             '“cam01,cam03,cam05”。留空则使用全部 7 个相机')
    return parser.parse_args()

args = parse_args()
INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir
#CAMERA_NAMES = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
_ALL_CAMERAS = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
if args.cams is None:
    CAMERA_NAMES = _ALL_CAMERAS
else:
    # 去重 + 保持输入顺序 + 过滤非法相机
    seen = set()
    user_cams = []
    for c in args.cams.split(','):
        c = c.strip()
        if c and c in _ALL_CAMERAS and c not in seen:
            seen.add(c)
            user_cams.append(c)
    if not user_cams:
        raise ValueError('--cams 未指定有效相机，请检查输入')
    CAMERA_NAMES = user_cams

print('本次处理的相机列表：', CAMERA_NAMES)
IMAGE_SIZE   = (1920, 1020)          # (W, H)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 工具 ----------
def to_device(x):
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)

# ---------- 统一内参读取 ----------
def read_intrinsic(cam_idx: int):
    """
    读取 save_calib() 保存的新格式：
    fx fy cx cy k1 k2 p1 p2 k3   （9 个数）
    返回 K(3×3), dist(5,)
    """
    vec = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt")
    if vec.ndim == 0:          # 单元素会退化成 0-d
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

# ---------- 工具：从图片文件名提取时间戳 ----------
def extract_timestamp_from_path(img_path: str) -> float:
    """
    输入:/abs/path/1509393489.574918.jpg
    输出:1509393489.574918
    """
    basename = os.path.basename(img_path)      # 1509393489.574918.jpg
    ts_str, _ = os.path.splitext(basename)     # 1509393489.574918
    return float(ts_str)


def box3d_iou(box1, box2):
    """
    计算两个3D框的IoU（只考虑xy平面，忽略z和角度）
    box: [x, y, z, l, w, h, yaw]
    """
    from shapely.geometry import Polygon

    def get_corners_xy(box):
        x, y, z, l, w, h, yaw = box
        cos, sin = np.cos(yaw), np.sin(yaw)
        dx = l / 2
        dy = w / 2
        corners = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ])
        rot = np.array([[cos, -sin], [sin, cos]])
        corners = (rot @ corners.T).T + [x, y]
        return Polygon(corners)

    p1 = get_corners_xy(box1)
    p2 = get_corners_xy(box2)
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / (union + 1e-6)

def mkdir_p(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

mkdir_p(OUTPUT_DIR,
        f"{OUTPUT_DIR}/images",
        f"{OUTPUT_DIR}/ego_pose",
        f"{OUTPUT_DIR}/intrinsics",
        f"{OUTPUT_DIR}/intrinsics_undistorted",
        f"{OUTPUT_DIR}/extrinsics",
        f"{OUTPUT_DIR}/lidar",
        f"{OUTPUT_DIR}/track",
        f"{OUTPUT_DIR}/dynamic_mask")

# ---------- 数据加载 ----------
json_file = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')][0]
with open(os.path.join(INPUT_DIR, json_file), 'r') as f:
    DATA = json.load(f)
FRAMES = DATA['frames']

# ---------- 1. 标定 ----------
'''
def save_calib():
    for i, cam in enumerate(tqdm(CAMERA_NAMES, desc="calib")):
        cam_data = DATA['calib'][cam]
        intrinsic = np.array(cam_data['cam_intrinsic'])
        extrinsic = np.array(cam_data['cam_to_velo'])
        np.savetxt(f"{OUTPUT_DIR}/intrinsics/{i}.txt", intrinsic)
        np.savetxt(f"{OUTPUT_DIR}/extrinsics/{i}.txt", extrinsic)
'''
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
# ---------- 复用点云投影 ----------
def project_lidar_to_image(pts, T_velo2cam, K, img_size=(1920, 1020)):
    """pts: (N,3)  ndarray -> (N,2) 像素"""
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
             (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int)

# ---------- 2. 去畸变 + 保存图像 + 保存新内参 ----------
'''
def undistort_image(img_path, cam_idx, dst_path):
    intrinsic = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt")
    dist = np.array(DATA['calib'][CAMERA_NAMES[cam_idx]]['distortion'])
    K = intrinsic
    D = dist
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    undist = cv2.undistort(img, K, D, None, new_K)
    cv2.imwrite(dst_path, undist)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h),centerPrincipalPoint=True)
    roi = (0, 0, w, h)
    # 保存新内参
    #np.savetxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt", new_K)
    np.savetxt(f"{OUTPUT_DIR}/intrinsics_undistorted/{cam_idx}.txt", new_K)
'''
def undistort_image(img_path, cam_idx, dst_path):
    K, dist = read_intrinsic(cam_idx)      # 唯一入口
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
    undist = cv2.undistort(img, K, dist, None, new_K)
    cv2.imwrite(dst_path, undist)
    # 保存去畸变后的 3×3 内参
    np.savetxt(f"{OUTPUT_DIR}/intrinsics_undistorted/{cam_idx}.txt", new_K)
def save_images_undistort():
    tasks = []
    for idx, frame in enumerate(FRAMES):
        frame_id = frame['frame_id']
        for cam_idx, cam in enumerate(CAMERA_NAMES):
            jpg = f"{INPUT_DIR}/{cam}/{frame_id}.jpg"
            png = f"{OUTPUT_DIR}/images/{idx:06d}_{cam_idx}.jpg"
            tasks.append((jpg, cam_idx, png))
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(tqdm(ex.map(lambda args: undistort_image(*args), tasks),
                  total=len(tasks), desc="undistort images"))

# ---------- 9. 生成 timestamps.json（基于真实文件名时间戳） ----------
def save_timestamps():
    role_map = {
        'cam01': 'FRONT',
        'cam03': 'FRONT_LEFT',
        'cam05': 'FRONT_RIGHT',
        'cam06': 'SIDE_LEFT',
        'cam07': 'SIDE_RIGHT',
        'cam08': 'SIDE_LEFT',   # 如果数据里只有一侧，可合并
        'cam09': 'SIDE_RIGHT',
    }
    timestamps = {role: {} for role in role_map.values()}

    for idx, frame in enumerate(tqdm(FRAMES, desc="timestamps")):
        frame_id = frame['frame_id']          # 原始字符串帧号
        idx_str  = f"{idx:06d}"               # 输出连续 6 位编号
        for cam in CAMERA_NAMES:
            img_path = f"{INPUT_DIR}/{cam}/{frame_id}.jpg"
            if not os.path.exists(img_path):
                continue
            ts = extract_timestamp_from_path(img_path)
            role = role_map[cam]
            timestamps[role][idx_str] = ts

    with open(f"{OUTPUT_DIR}/timestamps.json", 'w') as f:
        json.dump(timestamps, f, indent=2)
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

# ---------- 8. 相机位姿按“帧ID_相机编号.txt”保存 ----------
# ---------- 8. 相机位姿按“帧序号_相机编号.txt”保存 ----------
def save_cam_pose_per_frame():
    """
    把每一帧、每个相机的 T_cam2world 保存为
    {OUTPUT_DIR}/cam_pose_txt/<帧序号>_相机编号.txt
    帧序号从 000000 开始顺序编号
    """
    out_dir = f"{OUTPUT_DIR}/ego_pose"
    mkdir_p(out_dir)

    for idx, frame in enumerate(tqdm(FRAMES, desc="cam_pose_txt")):
        # 构造帧序号字符串，如 000000, 000001, ...
        frame_idx_str = f"{idx:06d}"

        # 自车 → 世界
        q = np.array(frame['pose'][:4])
        t = np.array(frame['pose'][4:])
        T_world2ego = np.eye(4)
        T_world2ego[:3, :3] = R.from_quat(q).as_matrix()
        T_world2ego[:3, 3] = t
        T_ego2world = np.linalg.inv(T_world2ego)

        for cam_i, cam in enumerate(CAMERA_NAMES):
            # 相机 → 自车（外参）
            T_cam2ego = np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{cam_i}.txt")
            # 相机 → 世界
            T_cam2world = T_ego2world @ T_cam2ego
            # 保存
            file_name = f"{out_dir}/{frame_idx_str}_{cam_i}.txt"
            np.savetxt(file_name, T_cam2world)
'''
# ---------- 5. 轨迹 ----------
def build_trajectory():
    from scipy.optimize import linear_sum_assignment

    trajectory = {}
    track_info = {}
    track_cam_vis = {}
    obj_ids = {}
    tid_counter = 0

    active_tracks = {}  # TID -> {'box': ..., 'frame_idx': ..., 'miss_cnt': int}
    MAX_MISS = 5

    def match_truth_boxes(tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        tid_list = list(tracks.keys())
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, tid in enumerate(tid_list):
            box1 = tracks[tid]['box']
            for j, det in enumerate(detections):
                box2 = det['box']
                dist = np.linalg.norm(np.array(box1[:3]) - np.array(box2[:3]))
                size_diff = np.abs(np.array(box1[3:6]) - np.array(box2[3:6])).sum()
                dyaw = abs(box1[6] - box2[6])
                dyaw = min(dyaw, 2 * np.pi - dyaw)

                if dist > 3.0:
                    cost = 1e5
                else:
                    cost = dist + 0.3 * size_diff + 0.5 * dyaw
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 2.5:
                matched.append((tid_list[r], c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
        return matched, unmatched_tracks, unmatched_dets

    for idx, frame in enumerate(tqdm(FRAMES, desc="truth-only tracking")):
        fid = f"{idx:06d}"
        track_info[fid] = {}
        track_cam_vis[fid] = {i: [] for i in range(len(CAMERA_NAMES))}

        if 'annos' not in frame:
            # 整帧无真值：全部记漏检，**不输出任何框**
            for tid in list(active_tracks.keys()):
                active_tracks[tid]['miss_cnt'] += 1
            continue

        names, boxes = frame['annos']['names'], frame['annos']['boxes_3d']
        curr_detections = [
            {'label': name.lower(), 'box': box}
            for name, box in zip(names, boxes)
        ]

        matched, unmatched_tracks, unmatched_dets = match_truth_boxes(
            {tid: {'box': tr['box']} for tid, tr in active_tracks.items()},
            curr_detections
        )

        # 1. 更新已匹配轨迹（**真值框**）
        for tid, det_idx in matched:
            det = curr_detections[det_idx]
            active_tracks[tid]['box'] = det['box']
            active_tracks[tid]['frame_idx'] = idx
            active_tracks[tid]['miss_cnt'] = 0

        # 2. 新建轨迹
        for det_idx in unmatched_dets:
            det = curr_detections[det_idx]
            tid = f"T{tid_counter:04d}"
            tid_counter += 1
            active_tracks[tid] = {
                'box': det['box'],
                'frame_idx': idx,
                'miss_cnt': 0,
            }

        # 3. 漏检计数 & 清理
        for ut_idx in unmatched_tracks:
            tid = list(active_tracks.keys())[ut_idx]
            active_tracks[tid]['miss_cnt'] += 1
        to_del = [tid for tid, tr in active_tracks.items() if tr['miss_cnt'] > MAX_MISS]
        for tid in to_del:
            del active_tracks[tid]

        # 4. **仅当本帧匹配成功**才写入全局 trajectory（真值-only）
        for tid, det_idx in matched:
            det = curr_detections[det_idx]
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
                    'deformable': False, 
                }
                obj_ids[tid] = tid

            pose = np.eye(4)
            yaw = box[6]
            rot = R.from_euler('z', yaw).as_matrix()
            pose[:3, :3] = rot
            pose[:3, 3] = box[:3]
            trajectory[tid]['poses_vehicle'].append(pose.astype(np.float32))
            trajectory[tid]['frames'].append(idx)
            trajectory[tid]['timestamps'].append(float(frame['frame_id']))

            # 相机可见性
            for cam_i in range(len(CAMERA_NAMES)):
                track_cam_vis[fid][cam_i].append(tid)

            # 真值 box 信息
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

    # 保存
    with open(f"{OUTPUT_DIR}/track/trajectory.pkl", 'wb') as f:
        pickle.dump(trajectory, f)
    with open(f"{OUTPUT_DIR}/track/track_info.pkl", 'wb') as f:
        pickle.dump(track_info, f)
    with open(f"{OUTPUT_DIR}/track/track_camera_visible.pkl", 'wb') as f:
        pickle.dump(track_cam_vis, f)
    with open(f"{OUTPUT_DIR}/track/track_ids.json", 'w') as f:
        json.dump(obj_ids, f, indent=2)

    return trajectory, track_cam_vis
'''
# ---------- 5. 轨迹 + 奇数帧插值 ----------
def build_trajectory():
    trajectory = {}
    track_info = {}
    track_cam_vis = {}
    obj_ids = {}
    tid_counter = 0

    active_tracks = {}  # TID -> {'box': ..., 'frame_idx': ..., 'miss_cnt': int, 'velocity': ...}
    MAX_MISS = 5

    def predict_box(prev_box, prev2_box, dt):
        """匀速预测：位置/角度/尺寸 线性外推"""
        dp = (np.array(prev_box[:3]) - np.array(prev2_box[:3])) / dt
        dyaw = (prev_box[6] - prev2_box[6])
        if dyaw > np.pi: dyaw -= 2*np.pi
        if dyaw < -np.pi: dyaw += 2*np.pi
        dyaw /= dt
        dlwh = (np.array(prev_box[3:6]) - np.array(prev2_box[3:6])) / dt

        new_box = prev_box.copy()
        new_box[:3] += dp * dt
        new_box[3:6] += dlwh * dt
        new_box[6] += dyaw * dt
        return new_box

    def match_truth_boxes(tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        tid_list = list(tracks.keys())
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, tid in enumerate(tid_list):
            box1 = tracks[tid]['box']
            for j, det in enumerate(detections):
                box2 = det['box']
                dist = np.linalg.norm(np.array(box1[:3]) - np.array(box2[:3]))
                size_diff = np.abs(np.array(box1[3:6]) - np.array(box2[3:6])).sum()
                dyaw = abs(box1[6] - box2[6])
                dyaw = min(dyaw, 2 * np.pi - dyaw)
                cost = dist + 0.3 * size_diff + 0.5 * dyaw
                cost_matrix[i, j] = cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 2.5:
                matched.append((tid_list[r], c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
        return matched, unmatched_tracks, unmatched_dets

    for idx, frame in enumerate(tqdm(FRAMES, desc="tracking + interp")):
        fid = f"{idx:06d}"
        track_info[fid] = {}
        track_cam_vis[fid] = {i: [] for i in range(len(CAMERA_NAMES))}

        is_truth_frame = ('annos' in frame)
        dt = 0.2  # 假设 5 Hz，两帧间隔 0.2 s

        if is_truth_frame:
            names, boxes = frame['annos']['names'], frame['annos']['boxes_3d']
            curr_detections = [{'label': name.lower(), 'box': box} for name, box in zip(names, boxes)]
            matched, unmatched_tracks, unmatched_dets = match_truth_boxes(
                {tid: {'box': tr['box']} for tid, tr in active_tracks.items()}, curr_detections)

            # 更新匹配
            for tid, det_idx in matched:
                box = curr_detections[det_idx]['box']
                active_tracks[tid]['box'] = box
                active_tracks[tid]['frame_idx'] = idx
                active_tracks[tid]['miss_cnt'] = 0
                # 计算速度
                if 'prev_box' in active_tracks[tid]:
                    prev = active_tracks[tid]['prev_box']
                    active_tracks[tid]['velocity'] = (np.array(box[:3]) - np.array(prev[:3])) / dt
                active_tracks[tid]['prev_box'] = box

            # 新建
            for det_idx in unmatched_dets:
                box = curr_detections[det_idx]['box']
                tid = f"T{tid_counter:04d}"
                tid_counter += 1
                active_tracks[tid] = {'box': box, 'frame_idx': idx, 'miss_cnt': 0, 'prev_box': box}

            # 清理
            for ut_idx in unmatched_tracks:
                tid = list(active_tracks.keys())[ut_idx]
                active_tracks[tid]['miss_cnt'] += 1
            to_del = [tid for tid, tr in active_tracks.items() if tr['miss_cnt'] > MAX_MISS]
            for tid in to_del:
                del active_tracks[tid]

        else:
            # ===== 奇数帧：预测 =====
            for tid in list(active_tracks.keys()):
                tr = active_tracks[tid]
                if 'prev_box' not in tr or 'frame_idx' not in tr:
                    tr['miss_cnt'] += 1
                    continue
                prev_idx = tr['frame_idx']
                if idx - prev_idx == 1 and prev_idx >= 1:
                    # 用前两帧真值做预测
                    prev_box = tr['box']
                    if 'prev2_box' in tr:
                        prev2_box = tr['prev2_box']
                        pred_box = predict_box(prev_box, prev2_box, dt)
                    else:
                        # 只有一帧，复制
                        pred_box = prev_box.copy()
                    active_tracks[tid]['box'] = pred_box
                    active_tracks[tid]['frame_idx'] = idx
                    active_tracks[tid]['prev2_box'] = prev_box
                else:
                    tr['miss_cnt'] += 1

            to_del = [tid for tid, tr in active_tracks.items() if tr['miss_cnt'] > MAX_MISS]
            for tid in to_del:
                del active_tracks[tid]

        # ===== 无论真假，把当前 active_tracks 写进全局 =====
        for tid, tr in active_tracks.items():
            box = tr['box']
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
                obj_ids[tid] = tid

            pose = np.eye(4)
            yaw = box[6]
            rot = R.from_euler('z', yaw).as_matrix()
            pose[:3, :3] = rot
            pose[:3, 3] = box[:3]
            trajectory[tid]['poses_vehicle'].append(pose.astype(np.float32))
            trajectory[tid]['frames'].append(idx)
            trajectory[tid]['timestamps'].append(float(frame['frame_id']))

            for cam_i in range(len(CAMERA_NAMES)):
                track_cam_vis[fid][cam_i].append(tid)

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

    # 保存
    with open(f"{OUTPUT_DIR}/track/trajectory.pkl", 'wb') as f:
        pickle.dump(trajectory, f)
    with open(f"{OUTPUT_DIR}/track/track_info.pkl", 'wb') as f:
        pickle.dump(track_info, f)
    with open(f"{OUTPUT_DIR}/track/track_camera_visible.pkl", 'wb') as f:
        pickle.dump(track_cam_vis, f)
    with open(f"{OUTPUT_DIR}/track/track_ids.json", 'w') as f:
        json.dump(obj_ids, f, indent=2)

    return trajectory, track_cam_vis
# ---------- 6. GPU 动态掩码 ----------

def save_dynamic_mask_gpu(trajectory, track_cam_vis):
    print(f"Dynamic mask on {DEVICE}")
    faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4),
             (2,3,7,6), (0,3,7,4), (1,2,6,5)]

    for idx, frame in enumerate(tqdm(FRAMES, desc="dynamic_mask")):
        fid = f"{idx:06d}"
        for cam_i in range(len(CAMERA_NAMES)):
            mask = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8)

            K, _ = read_intrinsic(cam_i)          # 只要 K
            T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{cam_i}.txt"))

            for tid in track_cam_vis[fid][cam_i]:
                if tid not in trajectory:
                    continue
                traj = trajectory[tid]
                if traj['stationary']:
                    continue
                if idx not in traj['frames']:
                    continue
                pose_idx = traj['frames'].index(idx)
                pose = traj['poses_vehicle'][pose_idx]
                l, w, h = traj['length'], traj['width'], traj['height']

                # 8 角点
                corners_obj = 0.5 * np.array([
                    [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
                    [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h]
                ])
                corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]

                # 投影
                uv = project_lidar_to_image(corners_v, T_velo2cam, K)
                if len(uv) < 8:
                    continue

                # 填面
                for f in faces:
                    cv2.fillPoly(mask, [uv[f, :].astype(np.int32)], color=255)

            cv2.imwrite(f"{OUTPUT_DIR}/dynamic_mask/{fid}_{cam_i}.jpg", mask)
def save_track_vis(trajectory, track_cam_vis):
    vis_frames = []
    cam_idx = 0
    #intr = np.loadtxt(f"{OUTPUT_DIR}/intrinsics/{cam_idx}.txt")
    K, _      = read_intrinsic(cam_idx) 
    T_velo2cam = np.linalg.inv(np.loadtxt(f"{OUTPUT_DIR}/extrinsics/{cam_idx}.txt"))

    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]

    for idx, frame in enumerate(tqdm(FRAMES, desc="track_vis_with_id")):
        img_path = f"{OUTPUT_DIR}/images/{idx:06d}_{cam_idx}.jpg"
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        fid = f"{idx:06d}"

        for tid in track_cam_vis[fid][cam_idx]:
            if tid not in trajectory:
                continue
            traj = trajectory[tid]
            if idx not in traj['frames']:
                continue
            pose_idx = traj['frames'].index(idx)
            pose = traj['poses_vehicle'][pose_idx]
            l, w, h = traj['length'], traj['width'], traj['height']

            corners_obj = 0.5 * np.array([
                [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
                [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h]
            ])
            corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]
            uv = project_lidar_to_image(corners_v, T_velo2cam, K)
            if len(uv) < 8:
                continue

            # 画红色框
            for i, j in edges:
                cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 0, 255), 2)

            # 画ID
            center = np.mean(uv, axis=0).astype(int)
            cv2.putText(img, tid, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        vis_frames.append(img[:, :, ::-1])

    imageio.mimwrite(f"{OUTPUT_DIR}/track/track_vis_red_id.mp4", vis_frames, fps=5)
# ---------- main ----------
def main():
    global FRAMES
    FRAMES = DATA['frames'][args.start_idx : args.end_idx]
    if not FRAMES:
        print("⚠️  指定帧区间无数据，请检查 --start_idx / --end_idx")
        return
    save_calib()
    
    save_images_undistort()
    save_lidar()
    save_poses()
    save_cam_pose_per_frame() 
    traj, tcv = build_trajectory()
    save_dynamic_mask_gpu(traj, tcv)
    save_track_vis(traj, tcv)
    save_timestamps() 
    print("✅ ONCE → StreetCraft (GPU+对齐修复) 完成！")
    

if __name__ == '__main__':
    main()