from tqdm import tqdm
import json
import os
import shutil
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import random
import open3d as o3d
from PIL import Image
from pathlib import Path
from plyfile import PlyData, PlyElement
from typing import NamedTuple
import yaml
from pyquaternion import Quaternion
from types import SimpleNamespace
import math
import pickle
import sys
import imageio
import open3d as o3d
# 获取当前文件（qirui_converter.py）所在目录的上一级目录，即项目根目录，从而可以from waymo_helpers import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from waymo_processor.waymo_helpers import load_track, load_calibration, load_ego_poses, get_object, \
    project_label_to_image, project_label_to_mask, draw_3d_box_on_img, opencv2camera
from utils.pcd_utils import storePly
from utils.box_utils import bbox_to_corner3d, inbbox_points
from bidict import bidict


camera_names = ['front_main','left_front','right_front','left_rear','right_rear']
camera_name_2_id = {c:i for i,c in enumerate(camera_names)}
# intrinsics = {
#     'front_main': {'w': 1749, 'h': 1060, 'fx': 7332.73165922591, 'fy': 7329.88657415548, 'cx': 1938.66601342433, 'cy': 1088.49275453309},
#     'left_front': {'w': 1544, 'h': 850, 'fx': 681.3060728743784, 'fy': 675.4923176157037, 'cx': 772.0,'cy': 425.0},
#     'right_front': {'w': 1533, 'h': 832, 'fx': 675.2360529825143, 'fy': 671.977010691605, 'cx': 766.5,'cy': 416.0},
#     'left_rear': {'w': 1566, 'h': 845, 'fx': 677.6129265997787, 'fy': 680.5259422516575, 'cx': 783.0,'cy': 422.5},
#     'right_rear': {'w': 1541, 'h': 849, 'fx': 680.1104967755755, 'fy': 673.4213417751372, 'cx': 770.5,'cy': 424.5},
# }
qirui_name_2_notr_name = {
    'front_main': 'FRONT'
    ,'left_front': 'FRONT_LEFT'
    ,'right_front': 'FRONT_RIGHT'
    ,'left_rear': 'SIDE_LEFT'
    ,'right_rear': 'SIDE_RIGHT'
}


def undistort_image(image, camera_matrix, distortion_coefficients):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))

    # 两种方法，可以选择一种使用
    # 1. 使用cv2.undistort函数
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)

    # # 2. 使用remapping
    # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix, (w, h), 5)
    # undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # 裁剪图像，去除黑边
    x, y, w, h = roi
    cx, cy = new_camera_matrix[0, 2], new_camera_matrix[1, 2]
    w, h = round(min(cx - x, x + w - cx) * 2), round(min(cy - y, y + h - cy) * 2)
    x, y = round(cx - w / 2), round(cy - h / 2)
    new_camera_matrix[0, 2], new_camera_matrix[1, 2] = w / 2, h / 2
    undistorted_image = undistorted_image[y:y + h, x:x + w]
    return undistorted_image, new_camera_matrix, w, h
    # return image, camera_matrix, image.shape[1], image.shape[0]


def extract_frame_infos(filepath):
    with open(filepath, 'r') as f:
        frame_infos = json.load(f)
        mapping = frame_infos['mapping']
        cam_name_2_sensor_id = {}
        for sensor_id, sensor_name in mapping.items():
            cam_name_2_sensor_id[sensor_name.replace(' ','_')] = sensor_id
        frame_infos['mapping'] = cam_name_2_sensor_id
        frame_infos['frames'] = frame_infos['frames']
        return frame_infos


def extract_camera_intrinsics(frame_infos):
    camera_intrinsics = {}
    distcoeffs = {}
    for cam in camera_names:
        sensor_id = frame_infos['mapping'][cam]
        intrinsic = frame_infos['calibration'][sensor_id]['intrinsic']
        distcoeff = frame_infos['calibration'][sensor_id]['distcoeff']
        camera_intrinsics[cam] = np.array(intrinsic)
        distcoeffs[cam] = np.array(distcoeff)
    for i, camera_name in enumerate(camera_names):
        intrinsic = camera_intrinsics[camera_name]
        cam_id = camera_name_2_id[camera_name]
        path = os.path.join(intrinsic_dir, str(cam_id) + '.txt')
        with open(path, 'w') as output:
            output.write(str(intrinsic[0][0]) + '\n')
            output.write(str(intrinsic[1][1]) + '\n')
            output.write(str(intrinsic[0][2]) + '\n')
            output.write(str(intrinsic[1][2]) + '\n')

    return camera_intrinsics, distcoeffs


def extract_camera_extrinsics(frame_infos):
    camera_exntrinsics = {}
    for cam in camera_names:
        sensor_id = frame_infos['mapping'][cam]
        extrinsic = frame_infos['calibration'][sensor_id]['extrinsic']
        # extrinsic = np.matmul(extrinsic, opencv2camera)
        camera_exntrinsics[cam] = np.array(extrinsic)

    for camera, ext in camera_exntrinsics.items():
        filename = str(camera_name_2_id[camera]) + '.txt'
        np.savetxt(os.path.join(extrinsic_dir, filename), ext, delimiter=' ')
    return camera_exntrinsics


def extract_images(frame_infos, intrinsics, distcoeffs, raw_dir, output_dir, dynamic_mask_dir):
    distorted_intrinsics = {}
    distorted_w = {}
    distorted_h = {}
    images = {}
    for i, frame in tqdm(enumerate(frame_infos['frames']), total=len(frame_infos['frames']), desc=f"Processing images"):
        for camera_name in camera_names:
            frame_name = frame['frame_name']
            sensor_id = frame_infos['mapping'][camera_name]
            filename = frame[sensor_id]
            # filename = filename.replace('.jpg', '_undist.jpg')
            source_path = os.path.join(raw_dir, frame_name, filename)
            img_name = f'{i:06d}' + '_' + str(camera_name_2_id[camera_name])
            dest_path = os.path.join(output_dir, img_name + '.jpg')
            # shutil.copy2(source_path, dest_path)
            image = cv2.imread(source_path)
            image, new_K, new_w, new_h = undistort_image(image, intrinsics[camera_name], distcoeffs[camera_name])
            distorted_intrinsics[camera_name] = new_K
            distorted_w[camera_name] = new_w
            distorted_h[camera_name] = new_h
            cv2.imwrite(dest_path, image)
            images[img_name] = image
            # image[:, :] = 0
            # cv2.imwrite(os.path.join(dynamic_mask_dir, f'{i:06d}' + '_' + str(camera_name_2_id[camera_name]) + '.jpg'), image)

    for i, camera_name in enumerate(camera_names):
        intrinsic = distorted_intrinsics[camera_name]
        cam_id = camera_name_2_id[camera_name]
        path = os.path.join(intrinsic_dir, str(cam_id) + '.txt')
        with open(path, 'w') as output:
            output.write(str(intrinsic[0][0]) + '\n')
            output.write(str(intrinsic[1][1]) + '\n')
            output.write(str(intrinsic[0][2]) + '\n')
            output.write(str(intrinsic[1][2]) + '\n')
    return distorted_intrinsics, distorted_w, distorted_h, images


def quaternion_yaw(q) -> float:
    """从四元数计算偏航角(绕Z轴的旋转角度)"""
    # w, x, y, z = q
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw


def dict_to_namespace(data):
    """递归将字典转换为 SimpleNamespace 对象"""
    if isinstance(data, dict):
        # 递归处理每个键值对
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in data.items()
        })
    elif isinstance(data, list):
        # 处理列表中的元素（若包含字典）
        return [dict_to_namespace(item) for item in data]
    else:
        # 普通值直接返回
        return data


def reset_information_for_trajectory(trajectory_info, ego_frame_poses):
    # reset information for trajectory
    # poses, stationary, symmetric, deformable
    for label_id in trajectory_info.keys():
        new_trajectory = dict()
        trajectory = trajectory_info[label_id]
        trajectory = dict(sorted(trajectory.items(), key=lambda item: item[0]))

        dims = []
        frames = []
        timestamps = []
        poses_vehicle = []
        poses_world = []
        speeds = []

        for frame_id, bbox in trajectory.items():
            label = bbox['label']
            dims.append([bbox['height'], bbox['width'], bbox['length']])
            frames.append(int(frame_id))
            timestamps.append(bbox['timestamp'])
            speeds.append(bbox['speed'])
            pose_vehicle = np.eye(4)
            pose_vehicle[:3, :3] = np.array([
                [math.cos(bbox['heading']), -math.sin(bbox['heading']), 0],
                [math.sin(bbox['heading']), math.cos(bbox['heading']), 0],
                [0, 0, 1]
            ])
            pose_vehicle[:3, 3] = np.array([bbox['center_x'], bbox['center_y'], bbox['center_z']])

            ego_pose = ego_frame_poses[int(frame_id)]
            pose_world = np.matmul(ego_pose, pose_vehicle)

            poses_vehicle.append(pose_vehicle.astype(np.float32))
            poses_world.append(pose_world.astype(np.float32))

        dims = np.array(dims).astype(np.float32)
        dim = np.max(dims, axis=0)
        poses_vehicle = np.array(poses_vehicle).astype(np.float32)
        poses_world = np.array(poses_world).astype(np.float32)
        actor_world_postions = poses_world[:, :3, 3]
        distance = np.linalg.norm(actor_world_postions[0] - actor_world_postions[-1])
        dynamic = np.any(np.std(actor_world_postions, axis=0) > 0.5) or distance > 2

        new_trajectory['label'] = label
        new_trajectory['height'], new_trajectory['width'], new_trajectory['length'] = dim[0], dim[1], dim[2]
        new_trajectory['poses_vehicle'] = poses_vehicle
        new_trajectory['timestamps'] = timestamps
        new_trajectory['frames'] = frames
        new_trajectory['speeds'] = speeds
        new_trajectory['symmetric'] = (label != 'pedestrian')
        new_trajectory['deformable'] = (label == 'pedestrian')
        new_trajectory['stationary'] = not dynamic

        trajectory_info[label_id] = new_trajectory


def draw_filled_3d_box_mask(img, vertices, fill_color=(255, 255, 255)):
    # 定义3D包围盒的6个面，每个面由4个顶点索引组成
    # 这里假设vertices的索引与标准3D bounding box顶点定义一致
    faces = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],  # 前面
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],  # 后面
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],  # 底面
        [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],  # 顶面
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],  # 左面
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]  # 右面
    ]

    # 逐个面绘制并填充
    for face in faces:
        # 获取当前面的4个顶点坐标，并转换为整数
        points = [tuple(map(int, vertices[vertex])) for vertex in face]

        # 将点转换为适合cv2.fillPoly的格式
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 填充多边形（面）
        cv2.fillPoly(img, [pts], fill_color)

    return img


def extract_dynamic_objs_and_draw_dynamic_masks(output_dir, dynamic_mask_dir, intrinsics, distorted_w, distorted_h, images, raw_filepath, track_dir):
    with open(raw_filepath, 'r') as f:
        raw_defines = json.load(f)
        track_info = dict()  # 以每个frame的一个bbox为单位 frame_id, track_id 记录LiDAR-synced和Camera_synced bboxes
        track_camera_visible = dict()  # 以每个camera的一个bbox为单位 frame_id, camera_id, track_id 记录这个camera看到了哪些物体
        trajectory_info = dict()  # 以每个track物体的一个bbox为单位 track_id, frame_id 记录LiDAR-synced boxes
        object_ids = dict()  # 每个物体的track_id对应一个数字 （track_id, object_id）之后streetgaussian训练时用的是object_id
        track_vis_imgs_0, track_vis_imgs_1, track_vis_imgs_2 = [], [], []
        cagetories = set()

        for i, frame in enumerate(raw_defines['frames']):
            track_info_cur_frame = dict()
            track_camera_visible_cur_frame = dict()
            timestamp = ts_from_frame_name(frame['frame_name'])
            frame_images = dict()
            # dynamic_masks = dict()
            for camera_name in camera_names:
                img = images[f'{i:06d}_{camera_name_2_id[camera_name]}']
                frame_images[camera_name_2_id[camera_name]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                track_camera_visible_cur_frame[camera_name_2_id[camera_name]] = []
                # dynamic_masks[camera_name_2_id[camera_name]] = np.zeros(img.shape, dtype=np.uint8)
            for obj in frame['annotated_info']['3d_city_object_detection_annotated_info']['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']:
                if obj['category'] == 'car':
                    obj_class = "vehicle"
                elif obj['category'] == 'person': # {'person', 'motorcycle', 'truck', 'bicycle', 'car', 'tricycle', 'bus'}
                    obj_class = "pedestrian"
                # elif obj.type == label_pb2.Label.Type.TYPE_SIGN:
                #     obj_class = "sign"
                # elif obj.type == label_pb2.Label.Type.TYPE_CYCLIST:
                #     obj_class = "cyclist"
                else:
                    obj_class = "misc"
                cagetories.add(obj['category'])

                speed = np.linalg.norm(obj['velocity'][0:2])

                label_id = str(obj['track_id'])

                # Add one label
                if label_id not in trajectory_info.keys():
                    trajectory_info[label_id] = dict()

                if label_id not in object_ids:
                    object_ids[label_id] = len(object_ids)

                track_info_cur_frame[label_id] = dict()

                # LiDAR-synced box
                lidar_synced_box = dict()
                lidar_synced_box['height'] = obj['size'][2]
                lidar_synced_box['width'] = obj['size'][1]
                lidar_synced_box['length'] = obj['size'][0]
                lidar_synced_box['center_x'] = obj['obj_center_pos'][0]
                lidar_synced_box['center_y'] = obj['obj_center_pos'][1]
                lidar_synced_box['center_z'] = obj['obj_center_pos'][2]
                quat = Quaternion(obj['obj_rotation'])
                lidar_synced_box['heading'] = quaternion_yaw(quat)
                lidar_synced_box['label'] = obj_class
                lidar_synced_box['speed'] = speed
                lidar_synced_box['timestamp'] = timestamp
                track_info_cur_frame[label_id]['lidar_box'] = lidar_synced_box
                trajectory_info[label_id][f'{i:06d}'] = lidar_synced_box

                # Camera-synced box
                if True: # obj.camera_synced_box.ByteSize():
                    camera_synced_box = dict()
                    camera_synced_box['height'] = obj['size'][2]
                    camera_synced_box['width'] = obj['size'][1]
                    camera_synced_box['length'] = obj['size'][0]
                    camera_synced_box['center_x'] = obj['obj_center_pos_cam'][0]
                    camera_synced_box['center_y'] = obj['obj_center_pos_cam'][1]
                    camera_synced_box['center_z'] = obj['obj_center_pos_cam'][2]
                    quat = Quaternion(obj['obj_rotation_cam'])
                    camera_synced_box['heading'] = quaternion_yaw(quat)
                    camera_synced_box['label'] = obj_class
                    camera_synced_box['speed'] = speed
                    track_info_cur_frame[label_id]['camera_box'] = camera_synced_box

                    c = math.cos(camera_synced_box['heading'])
                    s = math.sin(camera_synced_box['heading'])
                    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = rotz_matrix
                    obj_pose_vehicle[:3, 3] = np.array([camera_synced_box['center_x'], camera_synced_box['center_y'], camera_synced_box['center_z']])

                    camera_visible = []
                    for camera_name in camera_names:
                        fx, fy, cx, cy = intrinsics[camera_name][0,0], intrinsics[camera_name][1,1], intrinsics[camera_name][0,2], intrinsics[camera_name][1,2]
                        camera_calibration = {'width': distorted_w[camera_name], 'height': distorted_h[camera_name], 'extrinsic': {'transform': extrinsics[camera_name]}, 'intrinsic': [fx,fy,cx,cy]}
                        vertices, valid = project_label_to_image(
                            dim=[camera_synced_box['length'], camera_synced_box['width'], camera_synced_box['height']],
                            obj_pose=obj_pose_vehicle,
                            calibration=dict_to_namespace(camera_calibration), # 为了能调用waymo_helpers中的函数，用dict_to_namespace组装camera_calibration
                        )

                        # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                        # partial visible for the case when not all corners can be observed
                        if valid.any():
                            camera_visible.append(camera_name_2_id[camera_name])
                            track_camera_visible_cur_frame[camera_name_2_id[camera_name]].append(label_id)
                        if valid.all() and camera_name in ['front_main']:
                            vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                            draw_3d_box_on_img(vertices, frame_images[camera_name_2_id[camera_name]], color=(255,0,0))
                            # draw_filled_3d_box_mask(dynamic_masks[camera_name_2_id[camera_name]], vertices)
                            # cv2.putText(frame_images[camera_name_2_id[camera_name]], obj['category'] + '_' + str(label_id), vertices[0,0,0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0))
                            # cv2.imwrite(os.path.join(track_dir, f"{camera_name}.jpg"), frame_images[camera_name_2_id[camera_name]])

                    # print(f'At frame {frame_id}, label {label_id} has a camera-synced box visible from cameras {camera_visible}')
                else:
                    track_info_cur_frame[obj.id]['camera_box'] = None

            # for camera_name in camera_names:
            #     filename = f'{i:06d}_{camera_name_2_id[camera_name]}.jpg'
            #     cv2.imwrite(os.path.join(dynamic_mask_dir, filename), dynamic_masks[camera_name_2_id[camera_name]])

            track_info[f'{i:06d}'] = track_info_cur_frame
            track_camera_visible[f'{i:06d}'] = track_camera_visible_cur_frame

            # track_vis_img = np.concatenate([
            #     frame_images[0],
            #     frame_images[1],
            #     frame_images[2]], axis=1)
            track_vis_imgs_0.append(frame_images[0])
            # track_vis_imgs_1.append(frame_images[1])
            # track_vis_imgs_2.append(frame_images[2])
        print(f'==== cagetories: {cagetories}')

        ego_frame_poses, _ = load_ego_poses(output_dir)
        reset_information_for_trajectory(trajectory_info, ego_frame_poses)

        imageio.mimwrite(os.path.join(track_dir, "track_vis_0.mp4"), track_vis_imgs_0, fps=24)
        # imageio.mimwrite(os.path.join(track_dir, "track_vis_1.mp4"), track_vis_imgs_1, fps=24)
        # imageio.mimwrite(os.path.join(track_dir, "track_vis_2.mp4"), track_vis_imgs_2, fps=24)
        with open(os.path.join(track_dir, "track_info.pkl"), 'wb') as f:
            pickle.dump(track_info, f)

        # save track camera visible
        with open(os.path.join(track_dir, "track_camera_visible.pkl"), 'wb') as f:
            pickle.dump(track_camera_visible, f)

        # save trajectory
        with open(os.path.join(track_dir, "trajectory.pkl"), 'wb') as f:
            pickle.dump(trajectory_info, f)

        with open(os.path.join(track_dir, "track_ids.json"), 'w') as f:
            json.dump(object_ids, f, indent=2)


def pose_to_homogeneous_matrix(pose_x, pose_y, pose_z, w,x,y,z):
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, 3] = [pose_x, pose_y, pose_z]
    rotation_matrix = Rotation.from_quat([w,x,y,z]).as_matrix()
    homogeneous_matrix[:3, :3] = rotation_matrix
    return homogeneous_matrix


def create_nearest_pose_finder(ego_poses: dict):
    """
    创建一个查找最近时间戳的函数，首次调用时缓存排序结果

    参数:
        ego_poses: 字典，键为时间戳，值为主车轨迹

    返回:
        function: 可重复调用的查找函数
    """
    # 初始化缓存变量
    sorted_timestamps = None

    def find_nearest_pose(timestamp: float) -> tuple:
        nonlocal sorted_timestamps

        if not ego_poses:
            raise ValueError("ego_poses 字典为空")

        # 首次调用时排序并缓存
        if sorted_timestamps is None:
            sorted_timestamps = sorted(ego_poses.keys())

        n = len(sorted_timestamps)

        # 处理边界情况
        if timestamp <= sorted_timestamps[0]:
            return sorted_timestamps[0], ego_poses[sorted_timestamps[0]]
        if timestamp >= sorted_timestamps[-1]:
            return sorted_timestamps[-1], ego_poses[sorted_timestamps[-1]]

        # 二分查找确定最接近的两个时间戳
        left, right = 0, n - 1
        while left < right - 1:
            mid = (left + right) // 2
            if sorted_timestamps[mid] < timestamp:
                left = mid
            else:
                right = mid

        # 比较两个候选时间戳，返回更接近的一个
        if abs(sorted_timestamps[left] - timestamp) <= abs(sorted_timestamps[right] - timestamp):
            return sorted_timestamps[left], ego_poses[sorted_timestamps[left]]
        else:
            return sorted_timestamps[right], ego_poses[sorted_timestamps[right]]

    return find_nearest_pose


def ts_from_frame_name(frame_name):
    ts = frame_name.split('sample_')[1]
    ts = int(ts)
    ts = ts/1000
    return int(ts)


def extra_ego_poses(filepath, frame_infos):
    """
    主车的频率高，用相机的时间戳去找最接近的主车时间戳，作为对应的主车位姿
    """
    with open(filepath, 'r') as f:
        raw_defines = json.load(f)
        all_ego_poses = {}
        timestamps = []
        # 创建查找器函数（仅排序一次）
        find_pose = create_nearest_pose_finder(all_ego_poses)
        for raw_define in raw_defines:
            ts = int(float(raw_define['timestamp'])*1000)
            position = raw_define['pose']['position']
            orientation = raw_define['pose']['orientation']
            posx,posy,posz = position['x'], position['y'], position['z']
            w,x,y,z = orientation['qw'], orientation['qx'], orientation['qy'], orientation['qz']
            timestamps.append(ts)
            all_ego_poses[ts] = pose_to_homogeneous_matrix(posx,posy,posz,w,x,y,z)
        ego_poses = {} # 只需要取相机对应帧的主车轨迹。
        for frame in frame_infos['frames']:
            ts = ts_from_frame_name(frame['frame_name'])
            _, pose = find_pose(ts)
            # ts = int(ts / 1000) # 微秒转毫秒
            ego_poses[ts] = pose

        ego_poses_ndarray = np.array([pose for pose in ego_poses.values()])
        mean_pos = ego_poses_ndarray[:,:,3].mean(axis=0)
        for ts, pose in ego_poses.items():
            pose[:3,3] -= mean_pos[:3]

        for i, ts in enumerate(sorted(ego_poses.keys())):
            ego_pose = ego_poses[ts]
            filename = os.path.join(ego_pose_dir, f'{i:06d}' + '.txt')
            np.savetxt(filename, ego_pose, delimiter=' ')
            for camera, ext in extrinsics.items():
                filename = os.path.join(ego_pose_dir, f'{i:06d}_{camera_name_2_id[camera]}' + '.txt')
                np.savetxt(filename, ego_pose @ ext, delimiter=' ')

        timestamps = {'FRAME': {}}
        timestamps.update({c: {} for c in qirui_name_2_notr_name.values()})
        for i, ts in enumerate(sorted(ego_poses.keys())):
            timestamps['FRAME'][f'{i:06d}'] = ts / 1000
            for camera in extrinsics.keys():
                notr_cme_name = qirui_name_2_notr_name[camera]
                timestamps[notr_cme_name][f'{i:06d}'] = ts / 1000
        with open(os.path.join(output_dir, 'timestamps.json'), 'w') as file:
            json.dump(timestamps, file, indent=4)

        return ego_poses


def find_closest_timestamp(all_timestamps, ts):
    min_diff = float('inf')
    closest = None
    for value in all_timestamps:
        diff = abs(ts - value)
        if diff < min_diff:
            min_diff = diff
            closest = value
    return closest


def extract_point_clounds(frame_infos, ego_poses, lidar_2_imu, lidar_2_cameras, extrinsics, intrinsics, distorted_w, distorted_h, img_dir, lidar_depth_dir, lidar_background_dir, lidar_actor_dir):
    track_info, track_camera_visible, trajectory = load_track(output_dir)
    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []

    sorted_ego_poses = []
    for i, ts in enumerate(sorted(ego_poses.keys())):
        sorted_ego_poses.append(ego_poses[ts])
    print("Processing LiDAR data...")
    for i, frame in tqdm(enumerate(frame_infos['frames']), total=len(frame_infos['frames']), desc=f"Processing Lidar points"):
        frame_name = frame['frame_name']
        filename = frame['lidar0']
        raw_pcd_path = os.path.join(raw_dir, frame_name, filename)
        point_cloud = o3d.io.read_point_cloud(raw_pcd_path)
        xyzs = np.asarray(point_cloud.points)#[:,[0,2,1]]
        rgbs = np.zeros((len(point_cloud.points), 3), dtype=np.uint8)
        points_in_lidar = np.hstack([xyzs, np.ones((xyzs.shape[0], 1))])
        xyzs = (sorted_ego_poses[i] @ lidar_2_imu @ points_in_lidar.T).T
        xyzs = xyzs[:,:3]

        # Generate lidar depth and get pointcloud rgb
        for camera_name in camera_names:
            cam_id = camera_name_2_id[camera_name]
            img_name = f'{i:06d}' + '_' + str(camera_name_2_id[camera_name])
            img_path = os.path.join(img_dir, img_name + '.jpg')
            image = cv2.imread(img_path)[..., [2, 1, 0]].astype(np.uint8)
            h, w = image.shape[:2]

            depth_filename = os.path.join(lidar_depth_dir, f'{i:06d}_{cam_id}.npz')

            points_in_camera = np.dot(lidar_2_cameras[camera_name], points_in_lidar.T).T
            instrinsic = np.concatenate([intrinsics[camera_name], np.array([[0,0,0]]).T], axis=1)
            points_pixel = np.dot(instrinsic, points_in_camera.T).T
            uvs = points_pixel / (points_pixel[:, 2].reshape(-1, 1))
            u, v, d = uvs[:, 0].astype(int), uvs[:, 1].astype(int), points_in_camera[:, 2]

            valid_idx = np.where((u >= 0) & (u < w) & (v >= 0) & (v < h) & (d > 0))[0]
            u, v, depth = u[valid_idx], v[valid_idx], points_in_camera[:, 2][valid_idx]
            # u, v, depth = merge_conflict_pixels(u, v, depth)
            depth_map = np.full((h, w), 0., dtype=np.float32)
            depth_map[v, u] = depth
            valid_depth_pixels_mask = (depth_map > 1e-1)
            np.savez_compressed(depth_filename, mask=valid_depth_pixels_mask, value=depth_map[valid_depth_pixels_mask].reshape(-1))

            # try:
            #     if cam_id == 0 and i == 0:
            #         depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            #         highlight_color = [255, 0, 0]
            #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #         # image[500, :] = np.array(highlight_color, dtype=np.uint8)
            #         image[depth_map > 1e-1] = np.array(highlight_color, dtype=np.uint8)
            #         depth_img_name = f'{i:06d}' + '_' + str(camera_name_2_id[camera_name])
            #         depth_img_path = os.path.join(img_dir, depth_img_name + '_depth.jpg')
            #         cv2.imwrite(depth_img_path, image)
            # except:
            #     print(f'error in visualize depth of {image_filename}, depth range: {depth.min()} - {depth.max()}')

            # Colorize
            rgbs[valid_idx] = image[v, u]

        # masks = camera_id > 0
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f'{i:06d}']
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue

            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(i)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            # xyzs_actor = xyzs_homo @ np.linalg.inv(sorted_ego_poses[i]).T @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = (np.linalg.inv(pose_vehicle) @ np.linalg.inv(sorted_ego_poses[i]) @ xyzs_homo.T).T
            xyzs_actor = xyzs_actor[..., :3]

            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)

            actor_mask = np.logical_or(actor_mask, inbbox_mask)

            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = rgbs[inbbox_mask]
            # masks_inbbox = masks[inbbox_mask]
            actor_mask_for_ply = np.ones((xyzs_inbbox.shape[0], 1), dtype=bool)

            pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
            pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
            # pointcloud_actor[track_id]['mask'].append(masks_inbbox)

            # masks_inbbox = masks_inbbox[..., None]
            os.makedirs(os.path.join(lidar_actor_dir, track_id), exist_ok=True)
            ply_actor_path = os.path.join(lidar_actor_dir, track_id, f'{i:06d}.ply')
            try:
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, actor_mask_for_ply)
            except:
                pass  # No pcd

        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = np.ones((xyzs.shape[0], 1), dtype=bool)[~actor_mask] #np.ones((xyzs_background.shape[0], 1), dtype=bool)
        # masks_background = masks[~actor_mask]
        # masks_background = masks_background[..., None]
        ply_background_path = os.path.join(lidar_background_dir, f'{i:06d}.ply')

        storePly(ply_background_path, xyzs_background, rgbs_background, masks_background)

    # for track_id, pointcloud in pointcloud_actor.items():
    #     xyzs = np.concatenate(pointcloud['xyz'], axis=0)
    #     rgbs = np.concatenate(pointcloud['rgb'], axis=0)
    #     # masks = np.concatenate(pointcloud['mask'], axis=0)
    #     # masks = masks[..., None]
    #     ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, 'full.ply')
    #
    #     try:
    #         storePly(ply_actor_path_full, xyzs, rgbs, masks)
    #     except:
    #         pass  # No pcd


def mask_images(input_dir, output_masked_image_dir, output_mask_dir):
    # for camera_name in ['camera_front','front_main','left_rear','right_rear']:
    for camera_name in ['left_front','right_front']:
        dir = os.path.join(input_dir, camera_name)
        filnames = os.listdir(dir)
        if not os.path.exists(os.path.join(output_masked_image_dir, camera_name)):
            os.makedirs(os.path.join(output_masked_image_dir, camera_name))
        if not os.path.exists(os.path.join(output_mask_dir, camera_name)):
            os.makedirs(os.path.join(output_mask_dir, camera_name))
        for filname in filnames:
            image = cv2.imread(os.path.join(dir,filname))
            image[:,:,:] = (255,255,255)
            h,w,_ = image.shape
            if camera_name == 'front_main':
                image[h-100:h, w//2-100:w//2+100] = (0,0,0)
                cv2.imwrite(os.path.join(output_masked_image_dir, camera_name, filname), image)
                image[:,:] = (255,255,255)
                image[h-100:h, w//2-100:w//2+100] = (0,0,0)
                cv2.imwrite(os.path.join(output_mask_dir, camera_name, filname), image)
            elif camera_name == 'left_rear':
                image[0:100, 0:370] = (0,0,0)
                cv2.imwrite(os.path.join(output_masked_image_dir, camera_name, filname), image)
                image[:, :] = (255,255,255)
                image[0:100, 0:370] = (0,0,0)
                cv2.imwrite(os.path.join(output_mask_dir, camera_name, filname), image)
            elif camera_name == 'right_rear':
                image[0:100, w-370:w] = (0,0,0)
                cv2.imwrite(os.path.join(output_masked_image_dir, camera_name, filname), image)
                image[:, :] = (255,255,255)
                image[0:100, w-370:w] = (0,0,0)
                cv2.imwrite(os.path.join(output_mask_dir, camera_name, filname), image)
            elif camera_name == 'camera_front':
                image[h-440:h, 0:w] = (0,0,0)
                cv2.imwrite(os.path.join(output_masked_image_dir, camera_name, filname), image)
                image[:, :] = (255,255,255)
                image[h-440:h, 0:w] = (0,0,0)
                cv2.imwrite(os.path.join(output_mask_dir, camera_name, filname), image)
            else:
                image[:, :] = (255,255,255)
                cv2.imwrite(os.path.join(output_mask_dir, camera_name, filname), image)


def merge_conflict_pixels(u,v,d):
    coord_dict = {}
    for i in range(len(u)):
        coord = (u[i], v[i])
        if coord not in coord_dict or d[i] < coord_dict[coord]:
            coord_dict[coord] = d[i]

    result_u = np.array([k[0] for k in coord_dict.keys()])
    result_v = np.array([k[1] for k in coord_dict.keys()])
    result_d = np.array([v for k, v in coord_dict.items()])
    return result_u, result_v, result_d


def generate_dynamic_masks(extrinsics, intrinsics, distorted_w, distorted_h, output_dir):
    track_info, track_camera_visible, trajectory = load_track(output_dir)
    for frame, track_info_frame in track_info.items():
        track_camera_visible_cur_frame = track_camera_visible[frame]
        for cam, track_ids in track_camera_visible_cur_frame.items():
            dynamic_mask_name = f'{frame}_{cam}.jpg'
            dynamic_mask = np.zeros((distorted_h[camera_names[cam]], distorted_w[camera_names[cam]]), dtype=np.uint8).astype(np.bool_)

            # deformable_mask_name = f'{frame}_{cam}_deformable.png'
            deformable_mask = np.zeros((distorted_h[camera_names[cam]], distorted_w[camera_names[cam]]), dtype=np.uint8).astype(np.bool_)

            calibration_dict = dict()
            calibration_dict['extrinsic'] = extrinsics[camera_names[cam]]
            calibration_dict['intrinsic'] = intrinsics[camera_names[cam]]
            calibration_dict['height'] = distorted_h[camera_names[cam]]
            calibration_dict['width'] = distorted_w[camera_names[cam]]

            for track_id in track_ids:
                object_tracklet = trajectory[track_id]
                if object_tracklet['stationary']:
                    continue
                pose_idx = trajectory[track_id]['frames'].index(int(frame))
                pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]
                height, width, length = trajectory[track_id]['height'], trajectory[track_id]['width'], trajectory[track_id]['length']
                box_mask = project_label_to_mask(
                    dim=[length, width, height],
                    obj_pose=pose_vehicle,
                    calibration=None,
                    calibration_dict=calibration_dict,
                )

                dynamic_mask = np.logical_or(dynamic_mask, box_mask)
                if trajectory[track_id]['deformable']:
                    deformable_mask = np.logical_or(deformable_mask, box_mask)

            dynamic_mask_path = os.path.join(dynamic_mask_dir, dynamic_mask_name)
            cv2.imwrite(dynamic_mask_path, dynamic_mask.astype(np.uint8) * 255)


def load_lidar_2_cameras(raw_dir):
    lidar_2_cameras = {}
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2camera', 'lidar2frontmain.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        lidar_2_cameras['front_main'] = np.array(config['transform'])
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2camera', 'lidar2leftfront.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        lidar_2_cameras['left_front'] = np.array(config['transform'])
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2camera', 'lidar2rightfront.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        lidar_2_cameras['right_front'] = np.array(config['transform'])
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2camera', 'lidar2leftrear.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        lidar_2_cameras['left_rear'] = np.array(config['transform'])
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2camera', 'lidar2rightrear.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        lidar_2_cameras['right_rear'] = np.array(config['transform'])
    return lidar_2_cameras


def load_lidar_2_imu(raw_dir):
    with open(os.path.join(raw_dir, 'extrinsics', 'lidar2imu', 'lidar2imu.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        translation = config['transform']['translation']
        rotation = config['transform']['rotation']
        return pose_to_homogeneous_matrix(translation['x'],translation['y'],translation['z'],rotation['w'],rotation['x'],rotation['y'],rotation['z'])


if __name__ == '__main__':
    # with open(r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\processed\street_crafter_049\interpolated_track\track_info.pkl', 'rb') as f:
    #     track_info = pickle.load(f)
    #     track_info = track_info
    # with open(r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\processed\street_crafter_049\interpolated_track\track_camera_visible.pkl', 'rb') as f:
    #     track_camera_visible = pickle.load(f)
    #     track_camera_visible = track_camera_visible
    # with open(r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\processed\street_crafter_049\interpolated_track\trajectory.pkl', 'rb') as f:
    #     trajectory = pickle.load(f)
    #     trajectory = trajectory
    # from waymo_processor.waymo_helpers import load_track, load_camera_info
    # intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\processed\street_crafter_049')

    raw_dir = r'D:\Projects\3dgs_datas\dataset\qirui\raw'
    output_dir = r'D:\Projects\3dgs_datas\dataset\qirui\notr'
    img_dir = os.path.join(output_dir, 'images')
    dynamic_mask_dir = os.path.join(output_dir, 'dynamic_mask')
    ego_pose_dir = os.path.join(output_dir, 'ego_pose')
    intrinsic_dir = os.path.join(output_dir, 'intrinsics')
    extrinsic_dir = os.path.join(output_dir, 'extrinsics')
    track_dir = os.path.join(output_dir, 'track')
    lidar_depth_dir = os.path.join(output_dir, 'lidar', 'depth')
    lidar_background_dir = os.path.join(output_dir, 'lidar', 'background')
    lidar_actor_dir = os.path.join(output_dir, 'lidar', 'actor')
    lidar_cond_dir = os.path.join(output_dir, 'lidar', 'color_render')
    os.makedirs(dynamic_mask_dir, exist_ok=True)
    os.makedirs(ego_pose_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(intrinsic_dir, exist_ok=True)
    os.makedirs(extrinsic_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)
    os.makedirs(lidar_depth_dir, exist_ok=True)
    os.makedirs(lidar_background_dir, exist_ok=True)
    os.makedirs(lidar_actor_dir, exist_ok=True)
    os.makedirs(lidar_cond_dir, exist_ok=True)

    lidar_2_cameras = load_lidar_2_cameras(raw_dir)
    lidar_2_imu = load_lidar_2_imu(raw_dir)
    frame_infos = extract_frame_infos(os.path.join(raw_dir, 'info.json'))
    intrinsics, distcoeffs = extract_camera_intrinsics(frame_infos)
    extrinsics = extract_camera_extrinsics(frame_infos)
    ego_poses = extra_ego_poses(os.path.join(raw_dir, 'localization.json'), frame_infos)
    intrinsics, distorted_w, distorted_h, images = extract_images(frame_infos, intrinsics, distcoeffs, raw_dir, img_dir, dynamic_mask_dir)
    extract_dynamic_objs_and_draw_dynamic_masks(output_dir, dynamic_mask_dir, intrinsics, distorted_w, distorted_h, images, os.path.join(raw_dir, 'dynamic_obj', 'autolabel_10hz', 'clip_1746752396800.json'), track_dir)
    # generate_dynamic_masks(extrinsics, intrinsics, distorted_w, distorted_h, output_dir)
    # mask_images(r'D:\Projects\3dgs_datas\dataset\horizon\20240508/images', r'D:\Projects\3dgs_datas\dataset\horizon\20240508/mask_images', r'D:\Projects\3dgs_datas\dataset\horizon\20240508/masks')
    extract_point_clounds(frame_infos, ego_poses, lidar_2_imu, lidar_2_cameras, extrinsics, intrinsics, distorted_w, distorted_h, img_dir, lidar_depth_dir, lidar_background_dir, lidar_actor_dir)
