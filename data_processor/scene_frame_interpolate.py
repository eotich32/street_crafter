import cv2
import pickle
import math
import json
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from waymo_processor.waymo_helpers import load_track, load_camera_info, opencv2camera
import subprocess
from collections import defaultdict
from waymo_processor.waymo_helpers import project_label_to_image
from types import SimpleNamespace
import platform


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

camera_id_2_name = {
    0: "FRONT",
    1: "FRONT_LEFT",
    2: "FRONT_RIGHT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT"
}


def convert_2_uniform_list(timestamps, ego_frame_poses, ego_cam_poses, track_infos):
    num_original_frames = len(track_infos)
    all_obj_trajectories = [None] * num_original_frames
    num_cameras = len(ego_cam_poses)
    for frame_str in track_infos.keys():
        frame_id = int(frame_str)
        objects_this_frame = []
        # 添加主车信息（包含timestamp）
        objects_this_frame.append({'id': 'ego','pose': ego_frame_poses[frame_id],'timestamp': timestamps['FRAME'][frame_str]})
        for cam_id in range(num_cameras):
            # 添加相机信息（包含timestamp）
            objects_this_frame.append({'id': f'ego_cam_{cam_id}','pose': ego_cam_poses[cam_id][frame_id],'timestamp': timestamps[camera_id_2_name[cam_id]][frame_str]})

        other_objects = track_infos[frame_str]
        for obj_id, obj_info in other_objects.items():
            x_in_ego, y_in_ego, z_in_ego = obj_info['lidar_box']['center_x'], obj_info['lidar_box']['center_y'], \
                                           obj_info['lidar_box']['center_z']
            c = math.cos(obj_info['lidar_box']['heading'])
            s = math.sin(obj_info['lidar_box']['heading'])
            pose_in_ego = np.eye(4)
            pose_in_ego[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            pose_in_ego[:3, 3] = np.array([x_in_ego, y_in_ego, z_in_ego])
            pose_in_world = ego_frame_poses[frame_id] @ pose_in_ego
            objects_this_frame.append({
                'id': obj_id,
                'pose': pose_in_world,
                'timestamp': obj_info['lidar_box']['timestamp'],  # 添加timestamp
                'speed': obj_info['lidar_box']['speed']
            })
        all_obj_trajectories[frame_id] = objects_this_frame
    return all_obj_trajectories


def do_interpolation(timestamps, all_obj_trajectories, scene_dir, original_fps=10, target_fps=30):
    """
    主车、对手车轨迹插值，生成新的轨迹点，同时生成新的interpolated_images和interpolated_ego_pose目录
    """
    # 定义目录路径
    original_ego_pose_dir = os.path.join(scene_dir, 'ego_pose')
    original_image_dir = os.path.join(scene_dir, 'images')
    original_dynamic_mask_dir = os.path.join(scene_dir, 'dynamic_mask')
    original_sky_mask_dir = os.path.join(scene_dir, 'sky_mask')
    original_depth_dir = os.path.join(scene_dir, 'lidar/depth')
    original_background_ply_dir = os.path.join(scene_dir, 'lidar/background')
    original_actor_ply_dir = os.path.join(scene_dir, 'lidar/actor')
    interpolated_image_dir = os.path.join(scene_dir, 'interpolated_images')
    interpolated_ego_pose_dir = os.path.join(scene_dir, 'interpolated_ego_pose')
    interpolated_dynamic_mask_dir = os.path.join(scene_dir, 'interpolated_dynamic_mask')
    interpolated_sky_mask_dir = os.path.join(scene_dir, 'interpolated_sky_mask')
    interpolated_depth_dir = os.path.join(scene_dir, 'interpolated_lidar/depth')
    interpolated_background_ply_dir = os.path.join(scene_dir, 'interpolated_lidar/background')
    interpolated_actor_ply_dir = os.path.join(scene_dir, 'interpolated_lidar/actor')

    # 创建目录
    os.makedirs(interpolated_image_dir, exist_ok=True)
    os.makedirs(interpolated_ego_pose_dir, exist_ok=True)
    os.makedirs(interpolated_dynamic_mask_dir, exist_ok=True)
    os.makedirs(interpolated_sky_mask_dir, exist_ok=True)
    os.makedirs(interpolated_depth_dir, exist_ok=True)
    os.makedirs(interpolated_background_ply_dir, exist_ok=True)
    os.makedirs(interpolated_actor_ply_dir, exist_ok=True)

    # 1. 准备基础数据
    original_timestamps = _get_original_timestamps(timestamps)
    new_num_frames, new_timestamps = _generate_new_timestamps(original_timestamps, original_fps, target_fps)
    frame_mapping = _create_frame_mapping(new_num_frames, original_timestamps, original_fps, target_fps)

    # 2. 收集物体轨迹数据（包含timestamp）
    obj_data = _collect_object_data(all_obj_trajectories)

    # 3. 对每个物体进行插值处理
    new_all_obj_trajectories = [[] for _ in range(new_num_frames)]
    for obj_id, data in obj_data.items():
        _process_object_interpolation(
            obj_id, data, new_all_obj_trajectories,
            frame_mapping, original_fps, target_fps,
            original_ego_pose_dir, interpolated_ego_pose_dir,
            original_image_dir, interpolated_image_dir,
            original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
            original_sky_mask_dir, interpolated_sky_mask_dir,
            original_depth_dir, interpolated_depth_dir,
            original_background_ply_dir, interpolated_background_ply_dir,
            original_actor_ply_dir, interpolated_actor_ply_dir
        )

    return new_all_obj_trajectories, new_timestamps, frame_mapping


def _get_original_timestamps(timestamps):
    """提取原始时间戳并按帧号排序"""
    # 处理FRAME字段的时间戳
    frame_timestamps = {int(frame_id): ts for frame_id, ts in timestamps['FRAME'].items()}

    # 处理相机的时间戳
    camera_timestamps = {}
    for cam_id, cam_name in camera_id_2_name.items():
        if cam_name in timestamps:
            camera_timestamps[cam_id] = {int(frame_id): ts for frame_id, ts in timestamps[cam_name].items()}

    return {
        'frame': frame_timestamps,
        'camera': camera_timestamps
    }


def _generate_new_timestamps(original_timestamps, original_fps, target_fps):
    """生成插值后的时间戳字典，包含帧和相机的时间戳"""
    original_interval = 1.0 / original_fps
    target_interval = 1.0 / target_fps

    # 计算总时长和新帧数（基于主帧时间戳）
    frame_timestamps = original_timestamps['frame']
    sorted_original_frames = sorted(frame_timestamps.keys())
    start_ts = frame_timestamps[sorted_original_frames[0]]
    end_ts = frame_timestamps[sorted_original_frames[-1]]
    total_duration = end_ts - start_ts
    new_num_frames = int(total_duration / target_interval) + 1

    # 生成新的主帧时间戳
    new_timestamps = {
        'FRAME': {}
    }
    for i in range(new_num_frames):
        frame_id = str(i).zfill(6)
        new_timestamps['FRAME'][frame_id] = start_ts + i * target_interval

    # 生成各相机的时间戳（与主帧同步）
    for cam_id, cam_name in camera_id_2_name.items():
        new_timestamps[cam_name] = {}
        for i in range(new_num_frames):
            frame_id = str(i).zfill(6)
            new_timestamps[cam_name][frame_id] = start_ts + i * target_interval

    return new_num_frames, new_timestamps


def _create_frame_mapping(new_num_frames, original_timestamps, original_fps, target_fps):
    """创建新帧到原始帧的映射关系，基于插值间隔判断原始帧"""
    target_interval = 1.0 / target_fps
    frame_timestamps = original_timestamps['frame']
    sorted_original_frames = sorted(frame_timestamps.keys())
    original_ts_list = [frame_timestamps[frame] for frame in sorted_original_frames]
    start_ts = original_ts_list[0]

    # 计算插值倍数：每1个原始帧需要插入多少个新帧
    # 例如：原始10FPS→目标30FPS，倍数为3，即每1个原始帧后插入2个新帧
    interp_multiplier = round(target_fps / original_fps)
    # 原始帧在新帧序列中的间隔（每隔interp_multiplier帧有一个原始帧）
    original_frame_interval = interp_multiplier

    frame_mapping = {}
    for new_frame_idx in range(new_num_frames):
        new_ts = start_ts + new_frame_idx * target_interval

        # 找到最接近的原始帧（基于时间戳，用于插值参考）
        closest_original_idx = np.argmin(np.abs(np.array(original_ts_list) - new_ts))
        closest_original_frame = sorted_original_frames[closest_original_idx]
        closest_original_ts = original_ts_list[closest_original_idx]

        # 判断是否为原始帧：新帧索引是原始帧间隔的整数倍
        # 例如间隔为3时，新帧0、3、6...均为原始帧
        is_original = (new_frame_idx % original_frame_interval == 0)

        # 进一步验证：确保该原始帧在原始帧列表中真实存在
        if is_original:
            # 计算当前新帧对应的原始帧索引
            corresponding_original_idx = new_frame_idx // original_frame_interval
            # 检查该索引是否在有效范围内
            if corresponding_original_idx >= len(sorted_original_frames):
                is_original = False

        frame_mapping[new_frame_idx] = {
            'original_frame': closest_original_frame,
            'is_original': is_original,
            'timestamp': new_ts,
            't': (new_ts - closest_original_ts) / (1.0 / original_fps) if not is_original else 0.0
        }

    return frame_mapping


def _collect_object_data(all_obj_trajectories):
    """收集每个物体的轨迹数据（包含timestamp）"""
    obj_data = {}  # 格式: {obj_id: {frames: [], poses: [], speeds: [], timestamps: []}}

    for frame_idx, frame_data in enumerate(all_obj_trajectories):
        if frame_data is None:
            continue
        for obj in frame_data:
            obj_id = obj['id']
            if obj_id not in obj_data:
                obj_data[obj_id] = {
                    'frames': [],
                    'poses': [],
                    'speeds': [],
                    'timestamps': []  # 新增timestamp存储
                }
            obj_data[obj_id]['frames'].append(frame_idx)
            obj_data[obj_id]['poses'].append(obj['pose'])
            obj_data[obj_id]['speeds'].append(obj.get('speed', 0.0))
            obj_data[obj_id]['timestamps'].append(obj['timestamp'])  # 收集timestamp

    return obj_data


def _process_object_interpolation(obj_id, data, new_all_obj_trajectories,
                                  frame_mapping, original_fps, target_fps,
                                  original_ego_pose_dir, interpolated_ego_pose_dir,
                                  original_image_dir, interpolated_image_dir,
                                  original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
                                  original_sky_mask_dir, interpolated_sky_mask_dir,
                                  original_depth_dir, interpolated_depth_dir,
                                  original_background_ply_dir, interpolated_background_ply_dir,
                                  original_actor_ply_dir, interpolated_actor_ply_dir):
    """处理单个物体的插值计算"""
    frames = np.array(data['frames'])
    poses = np.array(data['poses'])
    speeds = np.array(data['speeds'])
    timestamps = np.array(data['timestamps'])  # 取出原始timestamp

    # 确定插值范围（首次出现到最后一次出现）
    start_ts = timestamps[0]
    end_ts = timestamps[-1]

    # 处理每个新帧
    for new_frame_idx, mapping in frame_mapping.items():
        new_ts = mapping['timestamp']

        # 检查是否在插值范围内
        if not obj_id.startswith('ego') and (new_ts < start_ts or new_ts > end_ts):
            continue

        _process_single_frame(
            new_frame_idx, mapping, obj_id, frames, poses, speeds, timestamps,
            new_all_obj_trajectories, original_ego_pose_dir, interpolated_ego_pose_dir,
            original_image_dir, interpolated_image_dir,
            original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
            original_sky_mask_dir, interpolated_sky_mask_dir,
            original_depth_dir, interpolated_depth_dir,
            original_background_ply_dir, interpolated_background_ply_dir,
            original_actor_ply_dir, interpolated_actor_ply_dir
        )


def _process_single_frame(new_frame_idx, mapping, obj_id, frames, poses, speeds, timestamps,
                          new_all_obj_trajectories, original_ego_pose_dir, interpolated_ego_pose_dir,
                          original_image_dir, interpolated_image_dir,
                          original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
                          original_sky_mask_dir, interpolated_sky_mask_dir,
                          original_depth_dir, interpolated_depth_dir,
                          original_background_ply_dir, interpolated_background_ply_dir,
                          original_actor_ply_dir, interpolated_actor_ply_dir):
    """处理单个新帧的插值计算，包含timestamp插值"""
    # 获取帧映射信息
    original_frame_idx = mapping['original_frame']
    is_original = mapping['is_original']
    t = mapping['t']

    # 找到相邻的原始帧索引
    left_idx = np.searchsorted(frames, original_frame_idx, side='right') - 1
    right_idx = left_idx + 1 if left_idx + 1 < len(frames) else left_idx

    # 判断是否为原始帧
    if is_original:
        interpolated_pose = poses[left_idx]# if left_idx == right_idx else poses[right_idx]
        interpolated_speed = speeds[left_idx]# if left_idx == right_idx else speeds[right_idx]
        interpolated_timestamp = timestamps[left_idx]# if left_idx == right_idx else timestamps[right_idx]
    else:
        # 执行插值计算
        interpolated_pose = _interpolate_pose(poses[left_idx], poses[right_idx], t)
        interpolated_speed = _interpolate_speed(obj_id, speeds[left_idx], speeds[right_idx], t)
        interpolated_timestamp = _interpolate_timestamp(timestamps[left_idx], timestamps[right_idx], t)  # 插值timestamp

    # 添加到新轨迹数据（包含timestamp）
    new_all_obj_trajectories[new_frame_idx].append({
        'id': obj_id,
        'is_original': is_original,
        'pose': interpolated_pose,
        'speed': interpolated_speed,
        'timestamp': interpolated_timestamp  # 新增timestamp字段
    })

    build_frame_file_links(
        obj_id, new_frame_idx, interpolated_pose,
        is_original, original_frame_idx, original_ego_pose_dir, interpolated_ego_pose_dir,
        original_image_dir, interpolated_image_dir,
        original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
        original_sky_mask_dir, interpolated_sky_mask_dir,
        original_depth_dir, interpolated_depth_dir,
        original_background_ply_dir, interpolated_background_ply_dir,
        original_actor_ply_dir, interpolated_actor_ply_dir
    )


def _interpolate_pose(left_pose, right_pose, t):
    """插值计算位姿矩阵（平移线性插值，旋转SLERP插值）"""
    # 分解位姿矩阵为平移和旋转
    left_translation = left_pose[:3, 3]
    right_translation = right_pose[:3, 3]

    left_rotation = R.from_matrix(left_pose[:3, :3])
    right_rotation = R.from_matrix(right_pose[:3, :3])

    # 插值平移（线性插值）
    interpolated_translation = left_translation + t * (right_translation - left_translation)

    # 插值旋转（手动实现SLERP，兼容低版本SciPy）
    interpolated_rotation = _slerp(left_rotation, right_rotation, t)

    # 组合成新的位姿矩阵
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rotation.as_matrix()
    interpolated_pose[:3, 3] = interpolated_translation

    return interpolated_pose


def _slerp(rot1, rot2, t):
    """
    手动实现球面线性插值（SLERP）
    参数:
        rot1: 起始旋转（Rotation对象）
        rot2: 目标旋转（Rotation对象）
        t: 插值权重（0~1）
    返回:
        插值后的旋转（Rotation对象）
    """
    # 将旋转转换为四元数（x, y, z, w）
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()

    # 计算四元数点积
    dot = np.dot(q1, q2)

    # 确保点积为正（取最短路径）
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # 防止数值不稳定（接近1时用线性插值）
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        q /= np.linalg.norm(q)
        return R.from_quat(q)

    # 计算旋转角度
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    # SLERP公式
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    q = s1 * q1 + s2 * q2

    return R.from_quat(q)


def _interpolate_speed(obj_id, left_speed, right_speed, t):
    """插值计算速度（主车和相机除外）"""
    if obj_id == 'ego' or obj_id.startswith('ego_cam_'):
        return 0.0  # 主车和相机不需要计算速度
    return left_speed + t * (right_speed - left_speed)


def _interpolate_timestamp(left_ts, right_ts, t):
    """插值计算timestamp（线性插值）"""
    return left_ts + t * (right_ts - left_ts)


def build_frame_file_links(obj_id, new_frame_idx, pose, is_original, original_frame_idx,
                                    original_ego_pose_dir, interpolated_ego_pose_dir,
                                    original_image_dir, interpolated_image_dir,
                                    original_dynamic_mask_dir, interpolated_dynamic_mask_dir,
                                    original_sky_mask_dir, interpolated_sky_mask_dir,
                                    original_depth_dir, interpolated_depth_dir,
                                    original_background_ply_dir, interpolated_background_ply_dir,
                                    original_actor_ply_dir, interpolated_actor_ply_dir):
    """处理主车和相机的文件生成与软链接，正确映射新旧帧号"""
    # 新帧号格式化（6位数字）
    new_frame_str = str(new_frame_idx).zfill(6)
    # 原始帧号格式化（6位数字）
    original_frame_str = str(original_frame_idx).zfill(6)

    # 生成文件名
    if obj_id == 'ego':
        new_ego_pose_filename = f"{new_frame_str}.txt"
        original_ego_pose_filename = f"{original_frame_str}.txt"
        new_depth_filename = f"{new_frame_str}.npz"
        original_depth_filename = f"{original_frame_str}.npz"
        new_background_ply_filename = f"{new_frame_str}.ply"
        original_background_ply_filename = f"{original_frame_str}.ply"
    elif obj_id.startswith('ego_cam_'):
        camera_id = obj_id.split('ego_cam_')[1]
        new_ego_pose_filename = f"{new_frame_str}_{camera_id}.txt"
        original_ego_pose_filename = f"{original_frame_str}_{camera_id}.txt"
        new_depth_filename = f"{new_frame_str}_{camera_id}.npz"
        original_depth_filename = f"{original_frame_str}_{camera_id}.npz"
        new_background_ply_filename = None
        original_background_ply_filename = None
    else:
        new_ego_pose_filename = None
        new_actor_ply_filename = f"{new_frame_str}.ply"
        original_actor_ply_filename = f"{original_frame_str}.ply"

    if new_ego_pose_filename is not None:
        dst_path = os.path.join(interpolated_ego_pose_dir, new_ego_pose_filename)
        dst_path = os.path.abspath(dst_path)
        np.savetxt(dst_path, pose)

    if obj_id.startswith('ego_cam_'):
        new_img_filename = new_ego_pose_filename.replace('.txt', '.png')
        original_img_filename = original_ego_pose_filename.replace('.txt', '.png')
        if is_original:
            create_soft_link_for_image(original_img_filename, new_img_filename, original_image_dir, interpolated_image_dir)
            create_soft_link_for_image(original_img_filename, new_img_filename, original_dynamic_mask_dir, interpolated_dynamic_mask_dir)
            create_soft_link_for_image(original_img_filename, new_img_filename, original_sky_mask_dir, interpolated_sky_mask_dir)
            create_soft_link_for_obj_prop_file(original_depth_filename, new_depth_filename, original_depth_dir, interpolated_depth_dir)
        else:
            # 生成占位符空文件，因为训练时加载文件统一把文件名与帧数关联，不方便修改
            with open(os.path.join(interpolated_image_dir, new_img_filename), 'w') as f:
                pass
            with open(os.path.join(interpolated_dynamic_mask_dir, new_img_filename), 'w') as f:
                pass
            with open(os.path.join(interpolated_sky_mask_dir, new_img_filename), 'w') as f:
                pass
    elif obj_id == 'ego':
        if is_original:
            create_soft_link_for_obj_prop_file(original_background_ply_filename, new_background_ply_filename, original_background_ply_dir, interpolated_background_ply_dir)
        # else:
        #     with open(os.path.join(interpolated_background_ply_dir, new_background_ply_filename), 'w') as f:
        #         pass
    else:
        # actor ply
        if is_original:
            original_actor_ply_dir = os.path.join(original_actor_ply_dir, obj_id)
            original_actor_path = os.path.join(original_actor_ply_dir, original_actor_ply_filename)
            if os.path.exists(original_actor_path):
                # 只处理相机可见物体
                interpolated_actor_ply_dir = os.path.join(interpolated_actor_ply_dir, obj_id)
                os.makedirs(interpolated_actor_ply_dir, exist_ok=True)
                create_soft_link_for_obj_prop_file(original_actor_ply_filename, new_actor_ply_filename, original_actor_ply_dir, interpolated_actor_ply_dir)


def create_soft_link_for_image(original_img_filename, new_img_filename, original_image_dir, interpolated_image_dir):
    src_img_path = os.path.join(original_image_dir, original_img_filename)
    dst_img_path = os.path.join(interpolated_image_dir, new_img_filename)
    src_img_path = os.path.abspath(src_img_path)
    dst_img_path = os.path.abspath(dst_img_path)
    if not os.path.exists(src_img_path):
        src_img_path = src_img_path.replace('png','jpg')
        dst_img_path = dst_img_path.replace('png','jpg')

    if not os.path.exists(dst_img_path):
        try:
            os.symlink(src_img_path, dst_img_path)
        except Exception as e:
            pass


def create_soft_link_for_obj_prop_file(original_filename, new_filename, original_dir, interpolated_dir):
    if original_filename is None or new_filename is None:
        return
    src_filepath = os.path.join(original_dir, original_filename)
    dst_filepath = os.path.join(interpolated_dir, new_filename)
    src_filepath = os.path.abspath(src_filepath)
    dst_filepath = os.path.abspath(dst_filepath)
    if not os.path.exists(dst_filepath):
        try:
            os.symlink(src_filepath, dst_filepath)
        except Exception as e:
            pass


def build_track_info(all_obj_trajectories, ego_frame_poses):
    track_info = {}
    total_frames = len(all_obj_trajectories)
    for frame_idx in range(total_frames):
        frame_key = f"{frame_idx:06d}"
        track_info[frame_key] = {}

    for frame_idx, frame_data in enumerate(all_obj_trajectories):
        frame_key = f"{frame_idx:06d}"
        if not frame_data:
            continue

        for obj in frame_data:
            obj_id = obj['id']
            if obj_id == 'ego' or obj_id.startswith('ego_cam_'):
                continue

            pose_in_world = obj['pose']
            pose_in_ego = np.linalg.inv(ego_frame_poses[frame_idx]) @ pose_in_world
            rel_translation = pose_in_ego[:3, 3]

            rotation_matrix = pose_in_ego[:3, :3]
            heading = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

            original_obj_info = original_obj_base_infos[obj_id]
            label = original_obj_info['label']

            box_info = {
                'width': original_obj_info['width'],
                'height': original_obj_info['height'],
                'length': original_obj_info['length'],
                'center_x': rel_translation[0],
                'center_y': rel_translation[1],
                'center_z': rel_translation[2],
                'heading': heading,
                'label': label,
                'speed': obj['speed'],
                'timestamp': obj['timestamp'],
            }

            # 同时添加lidar_box和camera_box（实际应用中可能不同）
            track_info[frame_key][obj_id] = {
                'lidar_box': box_info.copy(),
                'camera_box': box_info.copy()
            }
    return track_info


def build_trajectories(all_obj_trajectories, ego_frame_poses, original_obj_base_infos):
    trajectory = {}
    obj_frame_records = {}

    # 第一遍：收集基本信息
    for frame_idx, frame_data in enumerate(all_obj_trajectories):
        frame_key = f"{frame_idx:06d}"
        if not frame_data:
            continue

        for obj in frame_data:
            obj_id = obj['id']
            if obj_id == 'ego' or obj_id.startswith('ego_cam_'):
                continue
            if obj_id not in trajectory.keys():
                trajectory[obj_id] = {}

            if obj_id not in obj_frame_records.keys():
                obj_frame_records[obj_id] = []
            obj_frame_records[obj_id].append(frame_idx)

            pose_in_world = obj['pose']
            pose_in_ego = np.linalg.inv(ego_frame_poses[frame_idx]) @ pose_in_world

            original_obj_info = original_obj_base_infos[obj_id]
            label = original_obj_info['label']
            trajectory[obj_id][frame_key] = {
                'label': label,
                'width': original_obj_info['width'],
                'height': original_obj_info['height'],
                'length': original_obj_info['length'],
                'poses_vehicle': pose_in_ego,
                'frame': frame_idx,
                'timestamp': obj['timestamp'],
                'speed': obj['speed']
            }
    return trajectory, obj_frame_records


def fold_trajectories_2_statistic_info(trajectory, obj_frame_records):
    """
    输入是trajectory[obj_id][frame_id] = bbox的形式
    输出是trajectory[obj_id] = 信息合并到一起的形式
    """
    for obj_id in trajectory:
        # 获取所有出现的帧
        frames = sorted(obj_frame_records[obj_id])
        if not frames:
            continue

        # 收集所有位姿的平移部分（用于计算动态/静态）
        poses_translation = []
        widths = []
        heights = []
        lengths = []
        speeds = []
        timestamps = []
        poses_vehicle = []
        label = None
        for f in frames:
            frame_key = f"{f:06d}"
            pose = trajectory[obj_id][frame_key]['poses_vehicle']
            poses_translation.append(pose[:3, 3])
            widths.append(trajectory[obj_id][frame_key]['width'])
            heights.append(trajectory[obj_id][frame_key]['height'])
            lengths.append(trajectory[obj_id][frame_key]['length'])
            speeds.append(trajectory[obj_id][frame_key]['speed'])
            timestamps.append(trajectory[obj_id][frame_key]['timestamp'])
            poses_vehicle.append(trajectory[obj_id][frame_key]['poses_vehicle'])
            label = trajectory[obj_id][frame_key]['label']

        # 计算包围盒最大值
        max_width = max(widths) if widths else 0
        max_height = max(heights) if heights else 0
        max_length = max(lengths) if lengths else 0

        # 计算动态/静态（stationary）
        poses_np = np.array(poses_translation)
        std_dev = np.std(poses_np, axis=0).mean() if len(poses_np) > 1 else 0
        start_end_dist = np.linalg.norm(poses_np[0] - poses_np[-1]) if len(poses_np) > 1 else 0
        stationary = not (std_dev > 0.5 or start_end_dist > 2)

        trajectory[obj_id] = {}
        trajectory[obj_id]['width'] = max_width
        trajectory[obj_id]['height'] = max_height
        trajectory[obj_id]['length'] = max_length
        trajectory[obj_id]['stationary'] = stationary
        trajectory[obj_id]['frames'] = frames  # 完整帧列表
        trajectory[obj_id]['label'] = label
        trajectory[obj_id]['symmetric'] = label != 'pedestrian'
        trajectory[obj_id]['deformable'] = label == 'pedestrian'
        trajectory[obj_id]['timestamps'] = np.array(timestamps)
        trajectory[obj_id]['speeds'] = np.array(speeds)
        trajectory[obj_id]['poses_vehicle'] = np.array(poses_vehicle)
    return trajectory


def build_frame_visible_objects(all_obj_trajectories, track_info, extrinsics, intrinsics, cam_widths, cam_heights):
    track_camera_visible = {}
    camera_ids = camera_id_2_name.keys()

    for frame_idx, frame_data in enumerate(all_obj_trajectories):
        if not frame_data:
            continue

        track_camera_visible_cur_frame = dict()
        for cam_id in camera_ids:
            track_camera_visible_cur_frame[cam_id] = []
        for obj in frame_data:
            if obj['id'] == 'ego' or obj['id'].startswith('ego_cam_'):
                continue
            bbox = track_info[f'{frame_idx:06d}'][obj['id']]['lidar_box']
            for cam_id in camera_ids:
                fx, fy, cx, cy = intrinsics[cam_id][0, 0], intrinsics[cam_id][1, 1], intrinsics[cam_id][0, 2], intrinsics[cam_id][1, 2]
                camera_calibration = {'width': cam_widths[cam_id], 'height': cam_heights[cam_id],
                                      'extrinsic': {'transform': extrinsics[cam_id]}, 'intrinsic': [fx, fy, cx, cy]}
                c = math.cos(bbox['heading'])
                s = math.sin(bbox['heading'])
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array(
                    [bbox['center_x'], bbox['center_y'], bbox['center_z']])
                vertices, valid = project_label_to_image(
                    dim=[bbox['length'], bbox['width'], bbox['height']],
                    obj_pose=obj_pose_vehicle,
                    calibration=dict_to_namespace(camera_calibration),# 为了能调用waymo_helpers中的函数，用dict_to_namespace组装camera_calibration
                )

                # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                # partial visible for the case when not all corners can be observed
                if valid.any():
                    track_camera_visible_cur_frame[cam_id].append(obj['id'])
        track_camera_visible[f'{frame_idx:06d}'] = track_camera_visible_cur_frame
    return track_camera_visible


def save_as_notr_track_format(extrinsics, intrinsics, cam_widths, cam_heights, all_obj_trajectories, new_timestamps, frame_mapping, original_obj_base_infos, output_dir):
    total_frames = len(all_obj_trajectories)
    ego_frame_poses = [None]*total_frames
    for frame_idx, frame_data in enumerate(all_obj_trajectories):
        for obj in frame_data:
            obj_id = obj['id']
            if obj_id == 'ego':
                ego_frame_poses[frame_idx] = obj['pose']

    track_info = build_track_info(all_obj_trajectories, ego_frame_poses)
    trajectory, obj_frame_records = build_trajectories(all_obj_trajectories, ego_frame_poses, original_obj_base_infos)
    trajectory = fold_trajectories_2_statistic_info(trajectory, obj_frame_records)

    track_camera_visible = build_frame_visible_objects(all_obj_trajectories, track_info, extrinsics, intrinsics, cam_widths, cam_heights)

    with open(os.path.join(output_dir, 'timestamps.json'), 'w', encoding='utf-8') as f:
        json.dump(new_timestamps, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'frame_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(frame_mapping, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, 'track_info.pkl'), 'wb') as f:
        pickle.dump(dict(track_info), f)

    with open(os.path.join(output_dir, 'trajectory.pkl'), 'wb') as f:
        pickle.dump(dict(trajectory), f)

    with open(os.path.join(output_dir, 'track_camera_visible.pkl'), 'wb') as f:
        pickle.dump(dict(track_camera_visible), f)


def load_camera_widths_and_heights(scene_dir):
    cam_widths, cam_heights = [], []
    for cam_id in range(len(camera_id_2_name)):
        try:
            cam_img_path = os.path.join(scene_dir, 'images', f'000000_{cam_id}.png')
        except:
            cam_img_path = os.path.join(scene_dir, 'images', f'000000_{cam_id}.jpg')
        image = cv2.imread(cam_img_path)
        cam_widths.append(image.shape[1])
        cam_heights.append(image.shape[0])
    return cam_widths, cam_heights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print('Usage: scene_frame_interpolate.py --notr_scene_path=notr_scene_path')
    parser.add_argument('--notr_scene_path', default=r'D:\Projects\3dgs_datas\dataset\waymo\processed\street_crafter_049', help="notr scene path")
    args = parser.parse_args()

    timestamp_path = os.path.join(args.notr_scene_path, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)
    track_infos, track_camera_visible, trajectory = load_track(args.notr_scene_path, interpolated_first=False)
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(args.notr_scene_path)
    for i in range(len(extrinsics)):
        extrinsics[i] = np.matmul(extrinsics[i], np.linalg.inv(opencv2camera)) # TODO: waymo要这样，因为和waymo_helpers.py里get_extrinsic有关
    all_obj_trajectories = convert_2_uniform_list(timestamps, ego_frame_poses, ego_cam_poses, track_infos)
    original_fps = len(track_infos) / (max(timestamps['FRAME'].values()) - min(timestamps['FRAME'].values()))
    all_obj_trajectories, new_timestamps, frame_mapping = do_interpolation(timestamps, all_obj_trajectories, args.notr_scene_path, original_fps=original_fps, target_fps=30)
    output_dir = os.path.join(args.notr_scene_path, 'interpolated_track')
    os.makedirs(output_dir, exist_ok=True)

    original_obj_base_infos = {}
    for frame_str in track_infos:
        frame_objects = track_infos[frame_str]
        for obj_id, obj_info in frame_objects.items():
            if obj_id not in original_obj_base_infos:
                original_obj_base_infos[obj_id] = obj_info['lidar_box']

    cam_widths, cam_heights = load_camera_widths_and_heights(args.notr_scene_path)
    save_as_notr_track_format(extrinsics, intrinsics, cam_widths, cam_heights, all_obj_trajectories, new_timestamps, frame_mapping, original_obj_base_infos, output_dir)

