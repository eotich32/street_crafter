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
    camera_name = 'CAM_' + camera_name
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
    ego_cam_poses = [[] for i in range(5)]
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

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
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

def process_scene_lidar(nusc: NuScenes, scene_record: dict, save_dir_scene: str, nsweeps_lidar: int = 1):
    """
    Processes LiDAR data for a single nuScenes scene, similar to waymo_get_lidar_pcd.py.
    """
    #print(f"Processing scene {scene_record['name']} ({scene_record['token']})")
    #print(f"Saving to {save_dir_scene}")

    lidar_dir = os.path.join(save_dir_scene, 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)
    lidar_dir_background = os.path.join(lidar_dir, 'background')
    os.makedirs(lidar_dir_background, exist_ok=True)
    lidar_dir_actor = os.path.join(lidar_dir, 'actor')
    os.makedirs(lidar_dir_actor, exist_ok=True)
    lidar_dir_depth = os.path.join(lidar_dir, 'depth')
    os.makedirs(lidar_dir_depth, exist_ok=True)

    pointcloud_actor_tracks = {} 

    current_sample_token = scene_record['first_sample_token']
    frame_id_counter = 0 

    track_info, track_camera_visible, trajectory = load_track(save_dir_scene)

    #camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    num_samples_in_scene = scene_record['nbr_samples']
    pbar_samples = tqdm(total=num_samples_in_scene, desc=f"Scene {scene_record['name']}")

    while current_sample_token:
        sample_record = nusc.get('sample', current_sample_token)

        lidar_sd_token = sample_record['data']['LIDAR_TOP']
        lidar_sd_record = nusc.get('sample_data', lidar_sd_token)
        
        if nsweeps_lidar > 1 and lidar_sd_record['is_key_frame']:
            pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_record, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps_lidar)
        else: 
            lidar_pcl_path = nusc.get_sample_data_path(lidar_sd_token)
            pc = LidarPointCloud.from_file(lidar_pcl_path)

        points_lidar_sensor_frame = pc.points[:3, :].T
        num_pts = points_lidar_sensor_frame.shape[0]
        
        points_rgb = np.zeros((num_pts, 3), dtype=np.uint8) # Store as 0-255 uint8
        points_colored_this_frame_mask = np.zeros(num_pts, dtype=bool) # Mask for if a point has been colored by ANY camera

        for cam_idx, cam_channel in enumerate(camera_channels):
            cam_sd_token = sample_record['data'][cam_channel]
            cam_sd_record = nusc.get('sample_data', cam_sd_token)
            cam_path = nusc.get_sample_data_path(cam_sd_token)
            
            im = Image.open(cam_path)
            im_np = np.array(im) # H, W, 3 (RGB)
            img_h, img_w = im_np.shape[:2]

            pc_to_transform = LidarPointCloud(np.copy(pc.points))

            cs_lidar_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
            pc_to_transform.rotate(Quaternion(cs_lidar_record['rotation']).rotation_matrix)
            pc_to_transform.translate(np.array(cs_lidar_record['translation']))

            ego_pose_lidar = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
            pc_to_transform.rotate(Quaternion(ego_pose_lidar['rotation']).rotation_matrix)
            pc_to_transform.translate(np.array(ego_pose_lidar['translation']))

            ego_pose_cam = nusc.get('ego_pose', cam_sd_record['ego_pose_token'])
            pc_to_transform.translate(-np.array(ego_pose_cam['translation']))
            pc_to_transform.rotate(Quaternion(ego_pose_cam['rotation']).inverse.rotation_matrix)

            cs_cam_record = nusc.get('calibrated_sensor', cam_sd_record['calibrated_sensor_token'])
            pc_to_transform.translate(-np.array(cs_cam_record['translation']))
            pc_to_transform.rotate(Quaternion(cs_cam_record['rotation']).inverse.rotation_matrix)
            
            points_in_cam_sensor_frame = pc_to_transform.points[:3, :] 
            depths_in_cam_sensor_frame = points_in_cam_sensor_frame[2, :] 

            cam_intrinsics = np.array(cs_cam_record['camera_intrinsic'])
            points_2d_on_image = view_points(points_in_cam_sensor_frame, cam_intrinsics, normalize=True) 
            
            min_dist = 1.0
            projection_filter = (depths_in_cam_sensor_frame > min_dist) & \
                                (points_2d_on_image[0, :] >= 0) & (points_2d_on_image[0, :] < img_w) & \
                                (points_2d_on_image[1, :] >= 0) & (points_2d_on_image[1, :] < img_h)

            depth_map_for_cam = np.full((img_h, img_w), 0.0, dtype=np.float32) 
            
            u_coords_depth = points_2d_on_image[0, projection_filter].astype(int)
            v_coords_depth = points_2d_on_image[1, projection_filter].astype(int)
            depth_values_for_map = depths_in_cam_sensor_frame[projection_filter]

            sorted_indices = np.argsort(depth_values_for_map)
            u_sorted = u_coords_depth[sorted_indices]
            v_sorted = v_coords_depth[sorted_indices]
            d_sorted = depth_values_for_map[sorted_indices]

            for i in range(len(u_sorted)):
                 depth_map_for_cam[v_sorted[i], u_sorted[i]] = d_sorted[i]
            
            valid_depth_pixels_mask = (depth_map_for_cam > 1e-1) 
            depth_map_values = depth_map_for_cam[valid_depth_pixels_mask]
            
            depth_filename_npz = os.path.join(lidar_dir_depth, f"{frame_id_counter:06d}_{cam_idx}.npz")
            if depth_map_values.size > 0: 
                np.savez_compressed(depth_filename_npz, mask=valid_depth_pixels_mask, value=depth_map_values)

            u_coords_color = points_2d_on_image[0, projection_filter].astype(int)
            v_coords_color = points_2d_on_image[1, projection_filter].astype(int)
            original_indices_projectable = np.where(projection_filter)[0] 

            for idx_in_projectable_array, original_pt_idx in enumerate(original_indices_projectable):
                if not points_colored_this_frame_mask[original_pt_idx]:
                    u, v = u_coords_color[idx_in_projectable_array], v_coords_color[idx_in_projectable_array]
                    points_rgb[original_pt_idx] = im_np[v, u, :3] 
                    points_colored_this_frame_mask[original_pt_idx] = True
        
        pc_global = LidarPointCloud(np.copy(pc.points)) 
        cs_lidar_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pc_global.rotate(Quaternion(cs_lidar_record['rotation']).rotation_matrix)
        pc_global.translate(np.array(cs_lidar_record['translation'])) #车辆坐标系
        ego_pose_lidar = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
        pc_global.rotate(Quaternion(ego_pose_lidar['rotation']).rotation_matrix)
        pc_global.translate(np.array(ego_pose_lidar['translation'])) 
        points_global_frame = pc_global.points[:3, :].T 

        ann_tokens = sample_record['anns']
        actor_points_mask_for_frame = np.zeros(num_pts, dtype=bool) # Mask for points belonging to ANY actor this frame
        for ann_token in ann_tokens:
            ann_record = nusc.get('sample_annotation', ann_token)
            instance_token = ann_record['instance_token']
            pose_idx = trajectory[instance_token]['frames'].index(frame_id_counter)
            pose_vehicle = trajectory[instance_token]['poses_vehicle'][pose_idx]

            actor_instance_dir = os.path.join(lidar_dir_actor, instance_token)
            os.makedirs(actor_instance_dir, exist_ok=True)
            if instance_token not in pointcloud_actor_tracks:
                pointcloud_actor_tracks[instance_token] = {'xyz_global': [], 'rgb': []}

            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))
            current_actor_points_mask = points_in_box(box, points_global_frame.T) 
            actor_points_mask_for_frame |= current_actor_points_mask

            actor_pts_global_this_ann = points_global_frame[current_actor_points_mask]
            actor_colors_this_ann = points_rgb[current_actor_points_mask] # Use the colors derived from camera projection

            if actor_pts_global_this_ann.shape[0] > 0:
                actor_pts_local_this_ann = np.copy(actor_pts_global_this_ann)
                xyzs = actor_pts_local_this_ann
                xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1) 
                xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
                xyzs_actor = xyzs_actor[..., :3]
                actor_pts_local_this_ann = xyzs_actor # 替换

                ply_actor_frame_path = os.path.join(actor_instance_dir, f"{frame_id_counter:06d}.ply")
                # For storePly, mask should be (N,1) bool
                actor_mask_for_ply = np.ones((actor_pts_local_this_ann.shape[0],1), dtype=bool) # Points in actor pcd are actor points
                storePly(ply_actor_frame_path, actor_pts_local_this_ann, actor_colors_this_ann, actor_mask_for_ply)
                # 统一为局部坐标系
                pointcloud_actor_tracks[instance_token]['xyz_global'].append(actor_pts_local_this_ann)
                pointcloud_actor_tracks[instance_token]['rgb'].append(actor_colors_this_ann)

        background_points_mask_bool = ~actor_points_mask_for_frame # This is (N,) bool
        background_pts_global = points_global_frame[background_points_mask_bool]
        background_colors = points_rgb[background_points_mask_bool]
        
        ply_background_frame_path = os.path.join(lidar_dir_background, f"{frame_id_counter:06d}.ply")
        background_visibility_mask = points_colored_this_frame_mask[background_points_mask_bool].reshape(-1,1)

        ones_dim = np.ones((background_pts_global.shape[0], 1))
        pc_bkg = LidarPointCloud(np.hstack((background_pts_global, ones_dim)).T)
        pc_bkg.translate(-np.array(ego_pose_lidar['translation']))
        pc_bkg.rotate(Quaternion(ego_pose_lidar['rotation']).inverse.rotation_matrix)
        bkg_points_frame = pc_bkg.points[:3, :].T
        storePly(ply_background_frame_path, bkg_points_frame, background_colors, background_visibility_mask)

        current_sample_token = sample_record['next']
        frame_id_counter += 1
        pbar_samples.update(1)
    
    pbar_samples.close()

    #print(f"  Aggregating actor point clouds for scene {scene_record['name']}...")
    for instance_token, data in tqdm(pointcloud_actor_tracks.items(), desc="Aggregating Actors"):
        if not data['xyz_global']:
            continue
        all_actor_xyz_global = np.concatenate(data['xyz_global'], axis=0)
        all_actor_rgb = np.concatenate(data['rgb'], axis=0)
        
        if all_actor_xyz_global.shape[0] > 0:
            ply_actor_full_path = os.path.join(lidar_dir_actor, instance_token, 'full.ply')
            # full要统一局部坐标系
            full_actor_mask = np.ones((all_actor_xyz_global.shape[0], 1), dtype=bool)
            storePly(ply_actor_full_path, all_actor_xyz_global, all_actor_rgb, full_actor_mask)
    #print(f"Finished processing scene {scene_record['name']}")

time_stamp_dict = {"FRAME":{}, "FRONT_LEFT":{}, "FRONT_RIGHT":{}, "FRONT":{}, "SIDE_LEFT":{}, "SIDE_RIGHT":{}}
cam_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
cam_name_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

output_raw_dir = '/home/george/nuscene/nuscenes85'
data_path = "/home/george/nuscene/data/v1.0-trainval"
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

for scene_idx in tqdm(range(85, len(nusc.scene))):
    start_token = nusc.scene[scene_idx]['first_sample_token']
    scene_name = str(int(nusc.scene[scene_idx]['name'].split('-')[1])).zfill(3)
    output_scene_dir = os.path.join(output_raw_dir, scene_name)

    # 检查场景数据是否存在
    sample = nusc.get('sample', start_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_list[0]])
    ori_img_path = os.path.join(data_path, cam_data['filename'])
    if not os.path.exists(ori_img_path):
        #print(f"{ori_img_path} does not exist!")
        continue
    os.makedirs(output_scene_dir, exist_ok=True)

    # timestamp
    frame_idx = 0
    while start_token != '':
        sample = nusc.get('sample', start_token)
        #frame_idx加上前导0补全为6位
        frame_idx_str = str(frame_idx).zfill(6)
        time_stamp_dict['FRAME'][frame_idx_str] = process_time(sample['timestamp'])
        cam_front_data = nusc.get('sample_data', sample['data'][cam_list[0]])
        time_stamp_dict[cam_name_list[0]][frame_idx_str] = process_time(cam_front_data['timestamp'])
        cam_front_left_data = nusc.get('sample_data', sample['data'][cam_list[1]])
        time_stamp_dict[cam_name_list[1]][frame_idx_str] = process_time(cam_front_left_data['timestamp'])
        cam_front_right_data = nusc.get('sample_data', sample['data'][cam_list[2]])
        time_stamp_dict[cam_name_list[2]][frame_idx_str] = process_time(cam_front_right_data['timestamp'])
        cam_side_left_data = nusc.get('sample_data', sample['data'][cam_list[3]])
        time_stamp_dict[cam_name_list[3]][frame_idx_str] = process_time(cam_side_left_data['timestamp'])
        cam_side_right_data = nusc.get('sample_data', sample['data'][cam_list[4]])
        time_stamp_dict[cam_name_list[4]][frame_idx_str] = process_time(cam_side_right_data['timestamp'])
        start_token = sample['next']
        frame_idx += 1
    store_json_path = os.path.join(output_scene_dir, 'timestamps.json')
    json.dump(time_stamp_dict, open(store_json_path, 'w'), indent=4)

    # images
    start_token = nusc.scene[scene_idx]['first_sample_token']
    frame_idx = 0
    out_path = os.path.join(output_scene_dir, 'images')
    os.makedirs(out_path, exist_ok=True)
    while start_token != '':
        sample = nusc.get('sample', start_token)
        frame_idx_str = str(frame_idx).zfill(6)
        for i in range(len(cam_list)):
            cam_data = nusc.get('sample_data', sample['data'][cam_list[i]])
            ori_img_path = os.path.join(data_path, cam_data['filename'])
            img_path = os.path.join(out_path, f'{frame_idx_str}_{i}.png')
            cv2.imwrite(img_path, cv2.imread(ori_img_path))
        start_token = sample['next']
        frame_idx += 1

    # ego pose
    start_token = nusc.scene[scene_idx]['first_sample_token']
    frame_idx = 0
    ego_path = os.path.join(output_scene_dir, 'ego_pose')
    os.makedirs(ego_path, exist_ok=True)
    while start_token != '':
        sample = nusc.get('sample', start_token)
        frame_idx_str = str(frame_idx).zfill(6)
        for i in range(len(cam_list)):
            cam_data = nusc.get('sample_data', sample['data'][cam_list[i]])
            ego = nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_matrix = nuscenes_to_waymo(ego)
            ego_out_path = os.path.join(ego_path, f'{frame_idx_str}_{i}.txt')
            write_ego_pose(ego_out_path, ego_matrix)
            if i==0:
                ego_out_path = os.path.join(ego_path, f'{frame_idx_str}.txt')
                write_ego_pose(ego_out_path, ego_matrix)
        start_token = sample['next']
        frame_idx += 1

    # 内外参
    extrinsics_path = os.path.join(output_scene_dir, 'extrinsics')
    intrinsics_path = os.path.join(output_scene_dir, 'intrinsics')
    os.makedirs(extrinsics_path, exist_ok=True)
    os.makedirs(intrinsics_path, exist_ok=True)

    for i in range(len(cam_list)):
        cam_token = sample['data'][cam_list[i]]
        cam_data = nusc.get('sample_data', cam_token)
        sensor_token = cam_data['calibrated_sensor_token']
        cam_info = nusc.get('calibrated_sensor', sensor_token)

        cam_extrinsic_matrix = nuscenes_to_waymo(cam_info)
        out_ex_path = os.path.join(extrinsics_path, f"{i}.txt")
        write_ego_pose(out_ex_path, cam_extrinsic_matrix)

        out_in_path = os.path.join(intrinsics_path, f"{i}.txt")
        cam_intrinsic_list = nuscenes_to_waymo_intrinsics(cam_info['camera_intrinsic'])
        write_intrinsics(out_in_path, cam_intrinsic_list)

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

        for i in range(3):
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

            camera_visible = []
            camera_names = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT']
            for camera_name in camera_names:
                if check_anno_visible(start_token, camera_name, label):
                    camera_visible.append(camera_name)
                    if camera_name =='FRONT':
                        camera_id = 0
                    elif camera_name =='FRONT_LEFT':
                        camera_id = 1
                    else:
                        camera_id = 2
                    track_camera_visible_cur_frame[camera_id].append(label_id)
            
        track_info[f'{frame_idx:06d}'] = track_info_cur_frame    
        track_camera_visible[f'{frame_idx:06d}'] = track_camera_visible_cur_frame

        start_token = sample['next']
        frame_idx += 1

    ego_frame_poses, _ = load_ego_poses(output_scene_dir)
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

    track_dir = os.path.join(output_scene_dir, "track")
    os.makedirs(track_dir, exist_ok=True)
    # save track info
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

    # dynamic mask
    mask_dir = os.path.join(output_scene_dir, 'dynamic_mask')
    os.makedirs(mask_dir, exist_ok=True)
    viz = NuScenesVisualizer()
    start_token = nusc.scene[scene_idx]['first_sample_token']
    frame_idx = 0
    while start_token != '':
        sample = viz.nusc.get('sample', start_token)
        frame_idx_str = str(frame_idx).zfill(6)
        for i in range(len(cam_list)):
            sav_mask_path = os.path.join(mask_dir, f'{frame_idx_str}_{i}.png')
            viz.visualize_sample_2d(
                start_token, 
                camera_name=cam_list[i],
                save_mask=True,
                mask_dir=sav_mask_path
            )
            sav_mask_path = os.path.join(mask_dir, f'{frame_idx_str}_{i}_deformable.png') # deformable对应行人目标
            viz.visualize_sample_2d(
                start_token, 
                camera_name=cam_list[i],
                save_mask=True,
                mask_dir=sav_mask_path,
                category=1
            )
        start_token = sample['next']
        frame_idx += 1

    # lidar point cloud
    process_scene_lidar(nusc, nusc.scene[scene_idx], output_scene_dir)
