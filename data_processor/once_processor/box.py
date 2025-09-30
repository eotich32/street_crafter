#!/usr/bin/env python3
import numpy as np
import cv2
import pickle
import argparse
from pathlib import Path

IMAGE_SIZE = (1920, 1020)

# ---------- 复用点云投影 ----------
def project_lidar_to_image(pts, T_velo2cam, K, img_size=IMAGE_SIZE):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], 1)
    pts_cam = (T_velo2cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    w, h = img_size
    inside = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & \
             (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    return pts_2d[inside].astype(int), mask[inside]

# ---------- 3D 框 12 条棱的索引 ----------
EDGES = [
    # 底面 4 条
    [0, 1], [1, 2], [2, 3], [3, 0],
    # 顶面 4 条
    [4, 5], [5, 6], [6, 7], [7, 4],
    # 4 条竖直
    [0, 4], [1, 5], [2, 6], [3, 7]
]

COLOR_BOTTOM = (0, 255, 0)   # 底面绿色
COLOR_OTHER  = (160, 160, 160)  # 其余灰色

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    out, idx, cam_i = Path(args.output_dir), args.idx, args.cam
    fid = f"{idx:06d}"
    save_dir = out / "box3d_wireframe"
    save_dir.mkdir(exist_ok=True)

    with open(out / "track/trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    with open(out / "track/track_camera_visible.pkl", "rb") as f:
        track_cam_vis = pickle.load(f)

    K = np.loadtxt(out / "intrinsics" / f"{cam_i}.txt")
    T_velo2cam = np.linalg.inv(np.loadtxt(out / "extrinsics" / f"{cam_i}.txt"))

    img_path = out / "images" / f"{fid}_{cam_i}.jpg"
    if not img_path.exists():
        print(f"[WARN] image not found: {img_path}")
        return
    img = cv2.imread(str(img_path))

    for tid in track_cam_vis.get(fid, {}).get(cam_i, []):
        if tid not in trajectory:
            continue
        traj = trajectory[tid]
        pose = traj["poses_vehicle"][0]
        l, w, h = traj["length"], traj["width"], traj["height"]

        # 1. 8 个顶点（车体系）
        corners_obj = 0.5 * np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w,  h], [l, -w,  h], [l, w,  h], [-l, w,  h]
        ])
        corners_v = (pose[:3, :3] @ corners_obj.T).T + pose[:3, 3]

        # 2. 投影
        uv, valid_mask = project_lidar_to_image(corners_v, T_velo2cam, K)
        if uv is None or len(uv) < 8:
            continue

        # 3. 画 12 条边
        for i, (idx1, idx2) in enumerate(EDGES):
            color = COLOR_BOTTOM if i < 4 else COLOR_OTHER
            cv2.line(img, tuple(uv[idx1]), tuple(uv[idx2]), color, thickness=1)

    save_path = save_dir / f"3d_wireframe_{fid}_{cam_i}.jpg"
    cv2.imwrite(str(save_path), img)
    print(f"[INFO] saved: {save_path}")


if __name__ == "__main__":
    main()