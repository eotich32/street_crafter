import json
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

CAMERA_NAMES = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--once_json', required=True)
    p.add_argument('--image_root', required=True)
    p.add_argument('--frame_idx', type=int, default=100)
    p.add_argument('--cam_idx', type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.once_json, 'r') as f:
        data = json.load(f)
    frame = data['frames'][args.frame_idx]
    frame_id = frame['frame_id']
    cam = CAMERA_NAMES[args.cam_idx]

    img_path = f"{args.image_root}/{cam}/{frame_id}.jpg"
    img = cv2.imread(img_path)
    assert img is not None, f"找不到图像 {img_path}"

    calib = data['calib'][cam]
    K = np.array(calib['cam_intrinsic'])
    D = np.array(calib['distortion'])
    extr = np.array(calib['cam_to_velo'])  # cam->velo
    extr = np.linalg.inv(extr)             # velo->cam

    if 'annos' not in frame:
        print("本帧无标注")
        return
    box3d = frame['annos']['boxes_3d'][0]
    names = frame['annos']['names'][0]
    x, y, z, l, w, h, yaw = box3d
    print("【原始 box3d】 x,y,z,l,w,h,yaw =", box3d)

    # 1. 构建 8 角点（velo 系）
    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])
    rot = R.from_euler('z', yaw).as_matrix()
    corners = (rot @ corners.T).T + np.array([x, y, z])
    print("【velo 角点】\n", corners)

    # 2. 投影到图像
    corners_cam = (extr[:3, :3] @ corners.T + extr[:3, 3:4]).T
    pts_2d = (K @ corners_cam.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
    print("【图像角点】\n", pts_2d.astype(int))

    # 3. 画图
    for i, pt in enumerate(pts_2d.astype(int)):
        cv2.circle(img, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(img, str(i), tuple(pt+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        cv2.line(img, tuple(pts_2d[i].astype(int)), tuple(pts_2d[j].astype(int)), (0, 255, 0), 2)

    out = f"debug_numbers_f{args.frame_idx}_c{args.cam_idx}.jpg"
    cv2.imwrite(out, img)
    print("【保存】", out)

if __name__ == '__main__':
    main()