import os
import json
import math
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, points_in_box, transform_matrix
from pyquaternion import Quaternion
import cv2
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from plyfile import PlyData, PlyElement
def process_time(stamp):return stamp/1e6
def image_filename_to_cam(x): return int(x.split('.')[0][-1])
def image_filename_to_frame(x): return int(x.split('.')[0][:6])


notr_cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT', 'BACK']
time_stamp_dict = {"FRAME":{}}
for n in notr_cam_names:
    time_stamp_dict[n] = {}
nuscenes_cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
# data_path = r"D:\Projects\3dgs_datas\dataset\nuscenes\v1.0-mini"
# version='v1.0-mini'
data_path = '/mnt/data/dataset/nuscenes/nuScenes_raw/v1.0-trainval'
version='v1.0-trainval'

def quaternion_to_rotation_matrix(quat):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = quat
    
    # 归一化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    
    # 构建旋转矩阵
    rotation_matrix = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    
    return rotation_matrix

def nuscenes_to_waymo(nuscenes_data):
    """将nuScenes格式的ego_pose转换为Waymo格式"""
    # 提取旋转四元数和平移向量
    quat = nuscenes_data['rotation']  # [w, x, y, z]
    translation = nuscenes_data['translation']  # [x, y, z]
    
    # 转换四元数为旋转矩阵
    rotation_matrix = quaternion_to_rotation_matrix(quat)
    
    # 构建4x4齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def write_ego_pose(out_path, ego_matrix):
    with open(out_path, 'w') as f:
        for i in range(4):
            for j in range(4):
                f.write(f"{ego_matrix[i][j]:.18e}")
                if j != 3:
                    f.write(' ')
            f.write('\n')

def nuscenes_to_waymo_intrinsics(nuscenes_matrix, distortion_params=None):
    """将nuScenes内参矩阵转换为Waymo格式"""
    fx = nuscenes_matrix[0][0]
    fy = nuscenes_matrix[1][1]
    cx = nuscenes_matrix[0][2]
    cy = nuscenes_matrix[1][2]
    
    # 默认畸变参数（nuScenes未提供）
    if distortion_params is None:
        distortion_params = [0, 0, 0, 0, 0]  # k1, k2, p1, p2, k3
    
    return [fx, fy, cx, cy] + distortion_params

def write_intrinsics(out_path, intri_list):
    with open(out_path, 'w') as f:
        for intri in intri_list:
            f.write(f"{intri:.18e}")
            f.write('\n')

def check_anno_visible(sample_token, camera_name, ann):
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][camera_name])
    # 加载图像
    image_path = os.path.join(nusc.dataroot, cam_data['filename'])
    image = plt.imread(image_path)
    # 获取相机内参
    cam_intrinsic = np.array(nusc.get('calibrated_sensor', 
                                    cam_data['calibrated_sensor_token'])['camera_intrinsic'])
    # 获取ego pose信息
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    # 获取相机外参（从ego到相机的变换）
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_translation = np.array(cs_record['translation'])
    cam_rotation = Quaternion(cs_record['rotation'])
    #check
    # 获取标注框在全局坐标系下的信息
    box = Box(
        center=ann['translation'],
        size=ann['size'],
        orientation=Quaternion(ann['rotation']),
        name=ann['category_name']
    )
    # 坐标转换流程
    box.translate(-ego_translation)
    box.rotate(ego_rotation.inverse)
    box.translate(-cam_translation)
    box.rotate(cam_rotation.inverse)
    # 检查盒子是否在相机前方
    in_front = all(box.corners()[2, :] > 0.1)
    if not in_front:
        return False
    # 投影3D框到2D图像平面
    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
    # 计算2D边界框
    x_min = np.min(corners_2d[0])
    y_min = np.min(corners_2d[1])
    x_max = np.max(corners_2d[0])
    y_max = np.max(corners_2d[1])
    # 投影验证
    img_width, img_height = image.shape[1], image.shape[0]
    inside = (
        (x_min < img_width) and (x_max > 0) and
        (y_min < img_height) and (y_max > 0)
    )
    if inside:
        return True
    return False

def load_ego_poses(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    assert os.path.exists(ego_pose_dir), 'Ego pose dir not found.'
    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(len(notr_cam_names))]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(len(notr_cam_names))]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return ego_frame_poses, ego_cam_poses

def quaternion_yaw(q) -> float:
    """从四元数计算偏航角(绕Z轴的旋转角度)"""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw

class NuScenesVisualizer:
    def __init__(self):

        self.nusc = nusc
        
        self.vehicle_categories = [
            'vehicle.car',
            'vehicle.truck',
            'vehicle.bus',
            'vehicle.motorcycle',
            'vehicle.bicycle',
            'vehicle.construction'
        ]
        self.human_categories = ['human.pedestrian.adult']
        self.categories = [self.vehicle_categories + self.human_categories, self.human_categories]
    
    def visualize_sample_2d(self, sample_token, camera_name='CAM_FRONT', 
                           debug=False, save_mask=False, mask_dir=None, category=0):

        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][camera_name])
        
        # 加载图像
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = plt.imread(image_path)
        
        # 获取相机内参
        cam_intrinsic = np.array(self.nusc.get('calibrated_sensor', 
                                        cam_data['calibrated_sensor_token'])['camera_intrinsic'])
        
        # 获取ego pose信息
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        # 获取相机外参（从ego到相机的变换）
        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_translation = np.array(cs_record['translation'])
        cam_rotation = Quaternion(cs_record['rotation'])
        
        # 统计找到的车辆标注数量
        vehicle_annotations_found = 0
        valid_annotations = 0
        
        # 创建与原图大小一致的空白掩码图
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 定义3D框的所有平面
        box_faces = [
            [0, 1, 2, 3],  # 前平面
            [4, 5, 6, 7],  # 后平面
            [0, 3, 7, 4],  # 左平面
            [1, 2, 6, 5],  # 右平面
            [0, 1, 5, 4],  # 顶平面
            [2, 3, 7, 6]   # 底平面
        ]
        
        # 绘制车辆标注
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if ann['category_name'] in self.categories[category]:
                vehicle_annotations_found += 1
                
                # 获取标注框在全局坐标系下的信息
                box = Box(
                    center=ann['translation'],
                    size=ann['size'],
                    orientation=Quaternion(ann['rotation']),
                    name=ann['category_name'],
                    token=ann_token
                )
                
                # 坐标转换流程
                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)
                box.translate(-cam_translation)
                box.rotate(cam_rotation.inverse)
                
                # 检查盒子是否在相机前方
                in_front = all(box.corners()[2, :] > 0.1)
                if not in_front:
                    if debug:
                        print(f"警告: 标注 {ann_token} ({ann['category_name']}) 在相机后方")
                    continue
                
                # 投影3D框到2D图像平面
                corners_3d = box.corners()
                corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
                
                # 计算2D边界框
                x_min = np.min(corners_2d[0])
                y_min = np.min(corners_2d[1])
                x_max = np.max(corners_2d[0])
                y_max = np.max(corners_2d[1])
                
                # 投影验证
                img_width, img_height = image.shape[1], image.shape[0]
                inside = (
                    (x_min < img_width) and (x_max > 0) and
                    (y_min < img_height) and (y_max > 0)
                )
                
                if inside:
                    valid_annotations += 1
                    
                    # 生成掩码图
                    for face in box_faces:
                        # 提取平面顶点并转换为整数坐标
                        pts = np.array([corners_2d[:2, i].astype(int) for i in face], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        # 检查多边形是否有足够的顶点
                        if len(pts) >= 3:
                            # 在掩码图上填充多边形区域
                            cv2.fillPoly(mask, [pts], 255)
        
        # 保存掩码图
        if save_mask and mask_dir:
            # 保存掩码图
            cv2.imwrite(mask_dir, mask)
        
        # 输出调试信息
        if debug:
            print(f"样本 {sample_token} 在相机 {camera_name} 中共找到 {vehicle_annotations_found} 个车辆标注")
            print(f"其中 {valid_annotations} 个标注有效并显示在图像中")

def storePly(path, xyz, rgb, mask):
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)
    mask = mask.astype(np.bool_)
    # Define the dtype for the structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('mask', '?')
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb, mask), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def load_track(seq_save_dir):
    track_dir = os.path.join(seq_save_dir, 'track')
    assert os.path.exists(track_dir), f"Track directory {track_dir} does not exist."

    track_info_path = os.path.join(track_dir, 'track_info.pkl')
    with open(track_info_path, 'rb') as f:
        track_info = pickle.load(f)

    track_camera_visible_path = os.path.join(track_dir, 'track_camera_visible.pkl')
    with open(track_camera_visible_path, 'rb') as f:
        track_camera_visible = pickle.load(f)

    trajectory_path = os.path.join(track_dir, 'trajectory.pkl')
    with open(trajectory_path, 'rb') as f:
        trajectory = pickle.load(f)

    return track_info, track_camera_visible, trajectory


nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

scene_2_num_of_objects = {}


for scene_idx in tqdm(range(0, len(nusc.scene))):
    start_token = nusc.scene[scene_idx]['first_sample_token']
    scene_name = str(int(nusc.scene[scene_idx]['name'].split('-')[1])).zfill(3)

    # 检查场景数据是否存在
    sample = nusc.get('sample', start_token)
    cam_data = nusc.get('sample_data', sample['data'][nuscenes_cam_names[0]])

    # timestamp
    frame_idx = 0
    while start_token != '':
        sample = nusc.get('sample', start_token)
        #frame_idx加上前导0补全为6位
        frame_idx_str = str(frame_idx).zfill(6)
        time_stamp_dict['FRAME'][frame_idx_str] = process_time(sample['timestamp'])
        for i in range(len(notr_cam_names)):
            cam_front_data = nusc.get('sample_data', sample['data'][nuscenes_cam_names[i]])
            time_stamp_dict[notr_cam_names[i]][frame_idx_str] = process_time(cam_front_data['timestamp'])
        start_token = sample['next']
        frame_idx += 1

    # ego pose
    start_token = nusc.scene[scene_idx]['first_sample_token']
    frame_idx = 0
    ego_frame_poses = []
    while start_token != '':
        sample = nusc.get('sample', start_token)
        frame_idx_str = str(frame_idx).zfill(6)
        for i in range(len(nuscenes_cam_names)):
            cam_data = nusc.get('sample_data', sample['data'][nuscenes_cam_names[i]])
            ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_matrix = nuscenes_to_waymo(ego)
            if i==0:
                ego_frame_poses.append(ego_matrix)
        start_token = sample['next']
        frame_idx += 1

    # track box
    track_info = dict()
    track_camera_visible = dict()
    trajectory_info = dict()
    object_ids = dict()

    frame_idx = 0
    start_token = nusc.scene[scene_idx]['first_sample_token']
    while start_token != '':
        sample = nusc.get('sample', start_token)
        frame_idx_str = str(frame_idx).zfill(6)
        
        track_info_cur_frame = dict()
        track_camera_visible_cur_frame = dict()
        time_stamp = sample['timestamp'] / 1000000.0

        for i in range(len(notr_cam_names)):
            track_camera_visible_cur_frame[i] = []

        for label_token in sample['anns']:
            label = nusc.get('sample_annotation', label_token)
            instance_token = label['instance_token']
            if 'vehicle' in label['category_name']:
                obj_class = 'vehicle'
            elif 'pedestrian' in label['category_name']:
                obj_class = 'pedestrian'
            elif 'cyclist' in label['category_name']:
                obj_class = 'cyclist'
            else:
                obj_class = 'misc'
            
            #label_id = label_token
            label_id = instance_token # instance才能区分唯一物体 同一个物体可能有多个anno token
            if label_id not in trajectory_info.keys():
                trajectory_info[label_id] = dict()
            if label_id not in object_ids:
                object_ids[label_id] = len(object_ids)
            
            track_info_cur_frame[label_id] = dict()

            # LiDAR-synced box
            lidar_synced_box = dict()
            lidar_synced_box['height'] = label['size'][2]
            lidar_synced_box['width'] = label['size'][0] #1???
            lidar_synced_box['length'] = label['size'][1] #0???
            lidar_synced_box['center_x'] = label['translation'][0]
            lidar_synced_box['center_y'] = label['translation'][1]
            lidar_synced_box['center_z'] = label['translation'][2]
            quat = Quaternion(label['rotation'])
            # 计算绕Z轴的旋转角（heading）
            #_, _, yaw = quat.yaw_pitch_roll
            yaw = quaternion_yaw(quat)
            lidar_synced_box['heading'] = yaw
            lidar_synced_box['label'] = obj_class
            lidar_synced_box['speed'] = 0.0 # 后续再考虑是否需要计算
            lidar_synced_box['timestamp'] = time_stamp
            track_info_cur_frame[label_id]['lidar_box'] = lidar_synced_box                
            trajectory_info[label_id][f'{frame_idx:06d}'] = lidar_synced_box

            # Camera-synced box 因为nuScenes中没有该标注，默认全部为camera-synced box
            camera_synced_box = dict()
            camera_synced_box['height'] = label['size'][2]
            camera_synced_box['width'] = label['size'][0]
            camera_synced_box['length'] = label['size'][1]
            camera_synced_box['center_x'] = label['translation'][0]
            camera_synced_box['center_y'] = label['translation'][1]
            camera_synced_box['center_z'] = label['translation'][2]
            camera_synced_box['heading'] = yaw
            camera_synced_box['label'] = obj_class
            camera_synced_box['speed'] = 0.0
            track_info_cur_frame[label_id]['camera_box'] = camera_synced_box

            # 计算旋转矩阵
            c = math.cos(camera_synced_box['heading'])
            s = math.sin(camera_synced_box['heading'])
            rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            obj_pose_vehicle = np.eye(4)
            obj_pose_vehicle[:3, :3] = rotz_matrix
            obj_pose_vehicle[:3, 3] = np.array([camera_synced_box['center_x'], camera_synced_box['center_y'], camera_synced_box['center_z']])

            # camera_visible = []
            # for camera_id in range(len(notr_cam_names)):
            #     camera_name = notr_cam_names[camera_id]
            #     if check_anno_visible(start_token, nuscenes_cam_names[camera_id], label):
            #         camera_visible.append(camera_name)
            #         track_camera_visible_cur_frame[camera_id].append(label_id)
            
        track_info[f'{frame_idx:06d}'] = track_info_cur_frame    
        # track_camera_visible[f'{frame_idx:06d}'] = track_camera_visible_cur_frame

        start_token = sample['next']
        frame_idx += 1

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

    visible_obj_names = set()
    for name, obj in trajectory_info.items():
        if obj['label'] == 'vehicle' and not obj['stationary']:
            visible_obj_names.add(name)

    scene_2_num_of_objects[scene_name] = len(trajectory_info)
print(scene_2_num_of_objects)
sorted_items = sorted(scene_2_num_of_objects.items(), key=lambda x: x[1], reverse=False)
import csv
with open(os.path.join(data_path, 'num_objs.csv'), mode="w", newline="", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["scene_name", "num_dynamic_objs"])
    csv_writer.writerows(sorted_items)

