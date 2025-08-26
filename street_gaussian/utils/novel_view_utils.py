import numpy as np
import os
import math
import cv2
import copy
import torch
from tqdm import tqdm
from typing import List
from PIL import Image

from street_gaussian.datasets.base_readers import CameraInfo
from street_gaussian.utils.camera_utils import Camera, Camera4ForRenderCondition
from street_gaussian.utils.img_utils import visualize_depth_numpy, process_depth
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from data_processor.pandaset_processor.pandaset_helpers import get_lane_shift_direction as get_lane_shift_direction_pandaset
from data_processor.pandaset_processor.pandaset_helpers import LANE_SHIFT_SIGN as LANE_SHIFT_SIGN_PANDASET
from data_processor.waymo_processor.waymo_helpers import get_lane_shift_direction as get_lane_shift_direction_waymo
from data_processor.waymo_processor.waymo_helpers import LANE_SHIFT_SIGN as LANE_SHIFT_SIGN_WAYMO
from street_gaussian.config import cfg
from street_gaussian.utils.general_utils import matrix_to_quaternion_numpy, quaternion_to_matrix_numpy
from street_gaussian.pointcloud_processor.waymo_processor import WaymoPointCloudProcessor

from easyvolcap.utils.console_utils import *


def affine_inverse(A: np.ndarray):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return np.concatenate([np.concatenate([R.T, -R.T @ T], axis=-1), P], axis=-2)


# 四元数Slerp实现
def quaternion_slerp(q1, q2, t):
    """
    四元数球面线性插值
    q1, q2: 四元数 [w, x, y, z]
    t: 插值参数 [0, 1]
    """
    # 确保四元数方向一致
    if np.dot(q1, q2) < 0:
        q1 = -q1

    # 计算点积
    dot = np.dot(q1, q2)

    # 处理非常接近的四元数，避免数值不稳定
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # 确保点积在有效范围内
    dot = max(min(dot, 1.0), -1.0)

    # 计算夹角
    theta_0 = math.acos(dot)
    theta = theta_0 * t

    # 计算插值系数
    q3 = q2 - q1 * dot
    q3_norm = np.linalg.norm(q3)

    # 避免除以零
    if q3_norm < 1e-10:
        return q1

    q3 = q3 / q3_norm

    # 执行Slerp插值
    return q1 * math.cos(theta) + q3 * math.sin(theta)


def interpolate_novel_views(cameras, interpolation_points):
    novel_views = []
    skip_count = 0

    # 检查相机数量，至少需要2个相机才能进行插值
    if len(cameras) < 2:
        print("警告: 相机数量不足，无法进行插值")
        novel_views = cameras.copy()
    else:
        for i in range(len(cameras) - 1):
            # 获取相邻的两个相机位姿
            cam1 = cameras[i]
            cam2 = cameras[i + 1]

            # 获取位姿信息
            meta1 = cam1.metadata
            meta2 = cam2.metadata

            ego_pose1 = meta1['ego_pose'].copy()
            ego_pose2 = meta2['ego_pose'].copy()

            ext1 = meta1['extrinsic'].copy()
            ext2 = meta2['extrinsic'].copy()

            # 在两个位姿之间进行插值
            for j in range(1, interpolation_points + 1):
                t = j / (interpolation_points + 1)  # 插值比例

                # 线性插值生成新的ego_pose平移部分
                novel_ego_pose = ego_pose1.copy()
                novel_ego_pose[:3, 3] = (1 - t) * ego_pose1[:3, 3] + t * ego_pose2[:3, 3]

                # 使用四元数Slerp进行旋转插值
                rot1 = ego_pose1[:3, :3]
                rot2 = ego_pose2[:3, :3]

                # 转换旋转矩阵为四元数
                q1 = matrix_to_quaternion_numpy(rot1)
                q2 = matrix_to_quaternion_numpy(rot2)

                # 执行Slerp插值
                q_interp = quaternion_slerp(q1, q2, t)

                # 转换回旋转矩阵
                rot_interp = quaternion_to_matrix_numpy(q_interp)

                # 更新ego_pose中的旋转部分
                novel_ego_pose[:3, :3] = rot_interp

                # 插值外参
                novel_ext = (1 - t) * ext1 + t * ext2

                # 计算相机矩阵
                c2w = novel_ego_pose @ novel_ext
                RT = affine_inverse(c2w)
                R = RT[:3, :3].T
                T = RT[:3, 3]

                # 创建新的相机对象
                novel_view_camera = copy.copy(cam1)
                novel_view_camera = novel_view_camera._replace(metadata=copy.copy(cam1.metadata))

                # 生成新的图像名称
                image_name = cam1.image_name
                frame = meta1['frame']
                cam_id = meta1['cam']

                # 创建新视角路径
                tag = f'_interp_{i}_{j}'
                novel_view_dir = os.path.join(cfg.source_path, 'lidar', f'color_render{tag}')
                novel_view_image_name = f'{image_name}{tag}.png'

                metadata = novel_view_camera.metadata
                metadata['is_novel_view'] = True
                # metadata['novel_view_id'] = f'interp_{i}_{j}'
                metadata['ego_pose'] = novel_ego_pose
                metadata['extrinsic'] = novel_ext

                # 更新帧号为插值位置
                metadata['frame'] = int(frame + t * (meta2['frame'] - frame))

                novel_view_rgb_path = os.path.join(novel_view_dir, f'{str(metadata["frame"]).zfill(6)}_{cam_id}.png')
                novel_view_mask_path = os.path.join(novel_view_dir,
                                                    f'{str(metadata["frame"]).zfill(6)}_{cam_id}_mask.png')

                metadata['guidance_rgb_path'] = novel_view_rgb_path
                metadata['guidance_mask_path'] = novel_view_mask_path

                # 更新相机对象
                novel_view_camera = novel_view_camera._replace(
                    image_name=novel_view_image_name, R=R, T=T, guidance=dict(), metadata=metadata)

                novel_views.append(novel_view_camera)

        # 添加原始相机位姿
        novel_views.extend(cameras)
    return novel_views


def build_rotation_matrix(rotations):
    out = np.eye(3)
    for r in rotations:
        c, s = math.cos(r['rotate']), math.sin(r['rotate'])
        if r['axis'] == 'y':
            out = out.dot(np.array([
                [c, 0, -s],
                [0, 1, 0],
                [s, 0, c]
            ]))
        elif r['axis'] == 'x':
            out = out.dot(np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ]))
        else:  # z
            out = out.dot(np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ]))
    return out


def build_img_id(mode):
    tag = ''
    if mode['shift'] != 0: tag += f'_shift_{mode["shift"]:.2f}'
    for r in mode['rotations']:
        tag += f'_rotate_{r["axis"]}_{int(r["rotate"])}'
    return tag


def waymo_novel_view_cameras(cameras: List[CameraInfo], ego_frame_poses, obj_info, camera_tracklets):
    modes = []
    
    shifts = cfg.render.novel_view.shift if isinstance(cfg.render.novel_view.shift, list) else [cfg.render.novel_view.shift]
    if cfg.mode == 'train':
        shifts = [x for x in shifts if x != 0]
    for shift in shifts:
        modes.append({'shift': shift, 'rotations': [{'axis':'z', 'rotate': 0.0}]})
    # rotates = cfg.render.novel_view.rotate if isinstance(cfg.render.novel_view.rotate, list) else [cfg.render.novel_view.rotate]
    # rotates = [x for x in rotates if x != 0]
    # for rotate in rotates:
    #     modes.append({'shift': 0, 'rotate': rotate})
    # modes.append({'shift':50, 'custom':True, 'rotations': [{'axis': 'y','rotate': -90}]})
    # modes.append({'shift':50, 'custom':True, 'rotations': [{'axis': 'y','rotate': -45}]})
    # modes.append({'shift':20, 'custom':True, 'rotations': [{'axis': 'x','rotate': -45}, {'axis': 'y','rotate': -45}]})
    # modes.append({'shift':20, 'custom':True, 'rotations': [{'axis': 'x','rotate': 45}, {'axis': 'y','rotate': -45}]})

    novel_view_cameras = []
    skip_count = 0
    
    # cameras = [camera for camera in cameras if camera.metadata['cam'] in [0,1,2]]  # only consider the FRONT camera (whose cam_idx is marked as 0)
    pbar = tqdm(total=len(cameras) * len(modes), desc='Making novel view cameras')
    for i, mode in enumerate(modes):
        for camera in cameras:
            novel_view_camera = copy.copy(camera)
            novel_view_camera = novel_view_camera._replace(metadata=copy.copy(camera.metadata))

            image_name = novel_view_camera.image_name

            # make novel view path
            shift = mode['shift']
            tag = build_img_id(mode)
            novel_view_dir = os.path.join(cfg.source_path, 'lidar', f'color_render{tag}')
            novel_view_image_name = f'{image_name}{tag}.png'
            metadata = novel_view_camera.metadata
            metadata['is_novel_view'] = True
            metadata['ref_frame_image'] = novel_view_camera.image_path
            metadata['novel_view_id'] = tag
            cam, frame = metadata['cam'], metadata['frame']
            novel_view_rgb_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}.png')
            novel_view_mask_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}_mask.png')
            metadata['guidance_rgb_path'] = novel_view_rgb_path
            metadata['guidance_mask_path'] = novel_view_mask_path

            # make novel view camera
            ego_pose = metadata['ego_pose'].copy()
            ext = metadata['extrinsic'].copy()
            frame = metadata['frame']

            # shift
            if 'custom' in mode.keys():
                ego_pose[2, 3] += shift
            else:
                shift_direction = get_lane_shift_direction_waymo(ego_frame_poses, frame)
                scene_idx = os.path.split(cfg.source_path)[-1]
                ego_pose[:3, 3] += shift_direction * shift * LANE_SHIFT_SIGN_WAYMO[scene_idx]

            # rotate
            rot = build_rotation_matrix(mode['rotations'])
            ego_pose[:3, :3] = rot @ ego_pose[:3, :3]

            c2w = ego_pose @ ext
            RT = affine_inverse(c2w)
            R = RT[:3, :3].T
            T = RT[:3, 3]

            novel_view_camera = novel_view_camera._replace(
                image_name=novel_view_image_name,  R=R, T=T, guidance=dict(), metadata=metadata)
            novel_view_cameras.append(novel_view_camera)

            # TODO: fix obj_pose and sky and lidar
            cam = novel_view_camera.metadata['cam']
            frame_idx = novel_view_camera.metadata['frame_idx']

            skip_camera = False
            for obj_id in obj_info.keys():
                id = obj_info[obj_id]['id']
                if camera_tracklets[cam, frame_idx, id, -1] < 0.:
                    continue
                trans = camera_tracklets[cam, frame_idx, id, :3]
                view = (novel_view_camera.R).T @ trans + novel_view_camera.T
                depth = view[2]
                if depth < cfg.render.novel_view.train_actor_distance_thresh and \
                    depth > -cfg.render.novel_view.train_actor_distance_thresh:
                    skip_camera = True
                break

            skip_count += skip_camera
            novel_view_camera.metadata['skip_camera'] = skip_camera  # will skip camera for training if this is present
            pbar.update()

    novel_view_cameras = sorted(novel_view_cameras, key=lambda x: x.uid)
    log(f'Skipping {skip_count}/{len(novel_view_cameras)} novel view cameras')
    return novel_view_cameras


def append_interpolated_novel_view(novel_view_cam_infos, cameras, obj_info, camera_tracklets):
    lidar_condition_renderer = WaymoPointCloudProcessor()
    pbar = tqdm(total=len(cameras), desc='Making interpolated novel view cameras')
    for camera in cameras:
        novel_view_camera = copy.copy(camera)
        novel_view_camera = novel_view_camera._replace(metadata=copy.copy(camera.metadata))

        image_name = novel_view_camera.image_name

        # make novel view path
        tag = '_interpolated'
        novel_view_dir = os.path.join(cfg.source_path, 'lidar', f'color_render{tag}')
        novel_view_image_name = f'{image_name}{tag}.png'
        metadata = novel_view_camera.metadata
        metadata['is_novel_view'] = True
        metadata['novel_view_id'] = tag
        cam, frame = metadata['cam'], metadata['frame']
        novel_view_rgb_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}.png')
        novel_view_mask_path = os.path.join(novel_view_dir, f'{str(frame).zfill(6)}_{cam}_mask.png')
        metadata['guidance_rgb_path'] = novel_view_rgb_path
        metadata['guidance_mask_path'] = novel_view_mask_path

        camera_4_render = Camera4ForRenderCondition(
                     R=camera.R,
                     T=camera.T,
                     FoVx=camera.FovX,
                     FoVy=camera.FovY,
                     K=copy.deepcopy(camera.K),
                     image_height=camera.metadata['height'],
                     image_width=camera.metadata['width'],
                     metadata=camera.metadata,
                 )
        lidar_condition_renderer.render_condition(camera_4_render, obj_info)
        image = Image.open(camera.metadata['guidance_rgb_path']) # 实际diffusion的新视角不用到这个image，但是后面的代码结构，要求这个非空才能执行下去

        novel_view_camera = novel_view_camera._replace(
            image=image,
            image_name=novel_view_image_name, guidance=dict(), metadata=metadata)
        novel_view_cam_infos.append(novel_view_camera)

        # TODO: fix obj_pose and sky and lidar
        cam = novel_view_camera.metadata['cam']
        frame_idx = novel_view_camera.metadata['frame_idx']

        skip_camera = False
        for obj_id in obj_info.keys():
            id = obj_info[obj_id]['id']
            if camera_tracklets[cam, frame_idx, id, -1] < 0.:
                continue
            trans = camera_tracklets[cam, frame_idx, id, :3]
            view = (novel_view_camera.R).T @ trans + novel_view_camera.T
            depth = view[2]
            if depth < cfg.render.novel_view.train_actor_distance_thresh and \
                    depth > -cfg.render.novel_view.train_actor_distance_thresh:
                skip_camera = True
            break

        novel_view_camera.metadata['skip_camera'] = skip_camera  # will skip camera for training if this is present
        pbar.update()


def pandaset_novel_view_cameras(cameras: List[CameraInfo], cam_poses, obj_info, camera_tracklets):
    from street_gaussian.config import cfg
    shifts = cfg.render.novel_view.shift if isinstance(cfg.render.novel_view.shift, list) else [cfg.render.novel_view.shift]
    novel_view_cameras = []
    skip_count = 0
    pbar = tqdm(total=len(cameras) * len(shifts), desc='Making novel view cameras')

    for camera in cameras:
        if camera.metadata['cam'] != 0:
            continue  # only consider the FRONT camera (whose cam_idx is marked as 0)

        for i, shift in enumerate(shifts):
            novel_view_camera = copy.copy(camera)
            novel_view_camera = novel_view_camera._replace(metadata=copy.copy(camera.metadata))
            metadata = novel_view_camera.metadata
            metadata['is_novel_view'] = True
            metadata['novel_view_id'] = shift

            # make novel view path
            if shift == 0:
                novel_view_dir = os.path.join(cfg.source_path, 'lidar_forward', 'color_render')
            else:
                novel_view_dir = os.path.join(cfg.source_path, 'lidar_forward', f'color_render_shift_{shift:.2f}')
            cam, frame = metadata['cam'], metadata['frame']
            novel_view_rgb_path = os.path.join(novel_view_dir, f'{str(frame).zfill(3)}_{cam}.png')
            novel_view_mask_path = os.path.join(novel_view_dir, f'{str(frame).zfill(3)}_{cam}_mask.png')
            metadata['guidance_rgb_path'] = novel_view_rgb_path
            metadata['guidance_mask_path'] = novel_view_mask_path
            # make novel view camera

            c2w = metadata['extrinsic'].copy()
            cam = metadata['cam']
            frame = metadata['frame']

            # shift
            lane_shift_direction = get_lane_shift_direction_pandaset(cam_poses, cam, frame)

            scene_idx = os.path.split(cfg.source_path)[-1]

            c2w[:2, 3] += shift * lane_shift_direction[:2] * LANE_SHIFT_SIGN_PANDASET[scene_idx]

            # rotate
            # yaw = cfg.render.novel_view.rotate
            # c, s = math.cos(yaw), math.sin(yaw)
            # rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            # c2w[:3, :3] = rot @ c2w[:3, :3]

            metadata['extrinsic'] = c2w

            RT = affine_inverse(c2w)
            R = RT[:3, :3].T
            T = RT[:3, 3]

            novel_view_camera = novel_view_camera._replace(R=R, T=T, guidance=dict(), metadata=metadata)
            novel_view_cameras.append(novel_view_camera)

            # TODO: fix obj_pose and sky and lidar

            cam = novel_view_camera.metadata['cam']
            frame_idx = novel_view_camera.metadata['frame_idx']

            skip_camera = False
            for obj_id in obj_info.keys():
                id = obj_info[obj_id]['id']
                if camera_tracklets[cam, frame_idx, id, -1] < 0.:
                    continue
                trans = camera_tracklets[cam, frame_idx, id, :3]
                view = (novel_view_camera.R).T @ trans + novel_view_camera.T
                depth = view[2]
                if depth < cfg.render.novel_view.train_actor_distance_thresh and \
                        depth > -cfg.render.novel_view.train_actor_distance_thresh:
                    skip_camera = True
                    break

            skip_count += skip_camera
            novel_view_camera.metadata['skip_camera'] = skip_camera  # will skip camera for training if this is present
            pbar.update()

    novel_view_cameras = sorted(novel_view_cameras, key=lambda x: x.uid)
    log(f'Skipping {skip_count}/{len(novel_view_cameras)} novel view cameras')
    return novel_view_cameras


def virtual_wrap_kernel_numpy(
    tar_intrinsic,
    tar_extrinsic,
    tar_depth,
    tar_rgb,
    src_intrinsic,
    src_extrinsic,
    src_depth,
    src_rgb,
):
    tar_intrinsic = tar_intrinsic.cpu().numpy()
    tar_extrinsic = tar_extrinsic.cpu().numpy()
    tar_depth = tar_depth.cpu().numpy()
    tar_rgb = tar_rgb.cpu().numpy()
    src_intrinsic = src_intrinsic.cpu().numpy()
    src_extrinsic = src_extrinsic.cpu().numpy()
    src_depth = src_depth.cpu().numpy()
    src_rgb = src_rgb.cpu().numpy()

    h, w = tar_depth.shape[:2]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = tar_depth[v, u]
    # tar_depth_colored, _ = visualize_depth_numpy(tar_depth)
    # cv2.imwrite('tar_depth.png', tar_depth_colored)
    # src_depth_colored, _ = visualize_depth_numpy(src_depth)
    # cv2.imwrite('src_depth.png', src_depth_colored)
    # cv2.imwrite('src_rgb.png', cv2.cvtColor((src_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # __import__('ipdb').set_trace()

    points_pixel = np.stack([u, v, np.ones_like(u)], axis=-1) * z
    extrinsic_rel = affine_inverse(src_extrinsic) @ tar_extrinsic
    points_tar_pixel = points_pixel.reshape(-1, 3)
    points_tar_cam = points_tar_pixel @ np.linalg.inv(tar_intrinsic).T
    points_src_cam = points_tar_cam @ extrinsic_rel[:3, :3].T + extrinsic_rel[:3, 3].T
    points_src_depth = points_src_cam[:, 2]
    points_src_pixel = points_src_cam @ src_intrinsic.T
    points_src_pixel = points_src_pixel[:, :2] / points_src_pixel[:, 2:3]
    points_src_pixel = np.round(points_src_pixel).astype(np.int32)

    valid_depth = points_src_depth > 0
    valid_x = np.logical_and(points_src_pixel[:, 0] >= 0, points_src_pixel[:, 0] < w)
    valid_y = np.logical_and(points_src_pixel[:, 1] >= 0, points_src_pixel[:, 1] < h)
    project_mask = np.logical_and(valid_depth, np.logical_and(valid_x, valid_y))

    points_src_pixel = points_src_pixel[project_mask]
    points_src_depth_masked = points_src_depth[project_mask]
    points_src_depth_masked_query = src_depth[points_src_pixel[:, 1], points_src_pixel[:, 0], 0]
    occlusion_mask = points_src_depth_masked - points_src_depth_masked_query < 0.5
    points_src_pixel = points_src_pixel[occlusion_mask]

    mask = project_mask.copy()
    mask[project_mask] = occlusion_mask

    wrap_rgb = np.zeros_like(src_rgb).reshape(-1, 3)
    wrap_rgb_mask = src_rgb[points_src_pixel[:, 1], points_src_pixel[:, 0]]
    wrap_rgb[mask] = wrap_rgb_mask
    wrap_rgb = wrap_rgb.reshape(h, w, 3)
    mask = mask.reshape(h, w)

    # some visulization
    # if save_path is not None:
    #     cv2.imwrite(save_path, cv2.cvtColor((wrap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    # ax = axes[0, 0]
    # ax.imshow(tar_rgb)
    # ax.set_title('Target RGB')
    # ax = axes[0, 1]
    # ax.imshow(tar_depth_colored[..., [2, 1, 0]])
    # ax.set_title('Target Depth')
    # ax = axes[0, 2]
    # ax.imshow(src_rgb)
    # ax.set_title('Source RGB')
    # ax = axes[0, 3]
    # ax.imshow(src_depth_colored[..., [2, 1, 0]])
    # ax.set_title('Source Depth')
    # ax = axes[1, 0]
    # ax.imshow(mask)
    # ax.set_title('Mask')
    # ax = axes[1, 1]
    # ax.imshow(wrap_rgb)
    # ax.set_title('Wrap Target RGB')
    # masked_tar_rgb = tar_rgb.copy()
    # masked_tar_rgb[~mask] = 0
    # ax = axes[1, 2]
    # ax.imshow(masked_tar_rgb)
    # ax.set_title('Masked Target RGB')

    # diff = np.abs(wrap_rgb - masked_tar_rgb).sum(axis=-1)
    # diff_colored, _ = visualize_depth_numpy(diff, cmap=None)
    # ax = axes[1, 3]
    # ax.imshow(diff_colored)
    # ax.set_title('Diff Map')

    # plt.tight_layout()
    # plt.savefig(save_path)

    return wrap_rgb, mask


def virtual_wrap_kernel_torch(
    tar_intrinsic: torch.tensor,  # b, 3, 3
    tar_extrinsic: torch.tensor,  # b, 4, 4
    tar_depth: torch.tensor,  # b, h, w, 1
    tar_rgb: torch.tensor,  # b, h, w, 3
    src_intrinsic: torch.tensor,  # b, 3, 3
    src_extrinsic: torch.tensor,  # b, 4, 4
    src_depth: torch.tensor,  # b, h, w, 1
    src_rgb: torch.tensor,  # b, h, w, 3
):

    b, h, w = tar_depth.shape[:3]
    v, u = torch.meshgrid(
        torch.arange(h, dtype=torch.int32, device=tar_depth.device),
        torch.arange(w, dtype=torch.int32, device=tar_depth.device),
        indexing='ij')

    points_pixel = torch.stack([u, v, torch.ones_like(u)], axis=-1)
    points_pixel = points_pixel.unsqueeze(0).repeat(b, 1, 1, 1) * tar_depth

    extrinsic_rel = torch.einsum('bij,bjk->bik', torch.linalg.inv(src_extrinsic), tar_extrinsic)  # b, 4, 4
    points_tar_pixel = points_pixel.reshape(b, -1, 3)  # b, h * w, 3
    points_tar_cam = torch.einsum('bij,bjk->bik', points_tar_pixel, torch.linalg.inv(tar_intrinsic).permute(0, 2, 1))  # b, h * w, 3
    points_src_cam = torch.einsum('bij,bjk->bik', points_tar_cam, extrinsic_rel[:, :3, :3].permute(0, 2, 1)) + extrinsic_rel[:, :3, 3].unsqueeze(1)  # b, h * w, 3
    points_src_depth = points_src_cam[:, :, 2]  # b, h * w
    points_src_pixel = torch.einsum('bij,bjk->bik', points_src_cam, src_intrinsic.permute(0, 2, 1))  # b, h * w, 3
    points_src_pixel = points_src_pixel[:, :, :2] / points_src_pixel[:, :, 2:3]  # b, h * w, 2

    valid_depth = points_src_depth > 0
    valid_x = torch.logical_and(points_src_pixel[:, :, 0] >= 0, points_src_pixel[:, :, 0] < w)
    valid_y = torch.logical_and(points_src_pixel[:, :, 1] >= 0, points_src_pixel[:, :, 1] < h)
    project_mask = torch.logical_and(valid_depth, torch.logical_and(valid_x, valid_y))  # b, h * w

    trans_pos_norm = torch.zeros_like(points_src_pixel).float()
    trans_pos_norm[:, :, 0] = points_src_pixel[:, :, 0] / w
    trans_pos_norm[:, :, 1] = points_src_pixel[:, :, 1] / h
    trans_pos_norm = trans_pos_norm.detach() * 2. - 1.

    src_info = torch.cat([src_rgb, src_depth], dim=-1).permute(0, 3, 1, 2)  # b, 4, h, w (rgb + depth)
    wrap_info = torch.nn.functional.grid_sample(
        input=src_info,
        grid=trans_pos_norm.unsqueeze(1),
        mode='bilinear',
        align_corners=True,
        padding_mode='border')
    wrap_info = wrap_info.permute(0, 2, 3, 1).reshape(b, -1, 4)  # b, h * w, 4
    wrap_rgb = wrap_info[:, :, :3]
    wrap_depth = wrap_info[:, :, 3]

    depth_diff = torch.abs(wrap_depth - points_src_depth)
    depth_diff_thres = points_src_depth * 0.1
    occlusion_mask = depth_diff < depth_diff_thres  # b, h * w
    # occlusion_mask = depth_diff < 5 # b, h * w

    mask = torch.logical_and(project_mask, occlusion_mask)  # b, h * w

    wrap_rgb_full = torch.zeros_like(tar_rgb).reshape(b, -1, 3)  # b, h * w, 3
    wrap_rgb_full[mask] = wrap_rgb[mask]
    wrap_rgb_full = wrap_rgb_full.reshape(b, h, w, 3)  # b, h, w, 3
    mask = mask.reshape(b, h, w)  # b, h, w

    # wrap_rgb_full = wrap_rgb_full[0]
    # mask = mask[0]
    return wrap_rgb_full, mask


def lane_shift_directions(ego_frame_poses, frame):
    assert frame >= 0 and frame < len(ego_frame_poses)
    if frame == 0:
        ego_pose_delta = ego_frame_poses[1][:3, 3] - ego_frame_poses[0][:3, 3]
    else:
        ego_pose_delta = ego_frame_poses[frame][:3, 3] - ego_frame_poses[frame - 1][:3, 3]

    ego_pose_delta = ego_pose_delta[:2]  # x, y
    ego_pose_delta /= np.linalg.norm(ego_pose_delta)
    direction = np.array([ego_pose_delta[1], -ego_pose_delta[0], 0])  # y, -x
    return direction


def virtual_warp(
    src_camera: Camera,
    model: StreetGaussianModel,
    renderer: StreetGaussianRenderer,
):
    from street_gaussian.config import cfg
    novel_view_cfg = cfg.render.novel_view
    steps = novel_view_cfg.steps
    shift = novel_view_cfg.shift
    yaw = novel_view_cfg.rotate
    name = novel_view_cfg.name

    ego_frame_poses = model.metadata['ego_frame_poses']

    image_name = src_camera.image_name
    save_dir = os.path.join(cfg.model_path, 'virtual_wrap', name, image_name)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        src_result = renderer.render(src_camera, model)
        src_depth = process_depth(src_result['depth'], src_result['acc'])
        src_depth = src_depth.permute(1, 2, 0)
        src_rgb = src_camera.original_image.permute(1, 2, 0).to('cuda', non_blocking=True)
        src_extrinsic = src_camera.get_extrinsic()
        src_intrinsic = src_camera.get_intrinsic()
    src_noise = torch.randn_like(src_depth)

    tar_rgbs = []
    tar_depths = []
    tar_extrinsics = []
    tar_intrinsics = []

    for i, r in tqdm(enumerate(np.linspace(0, 1, steps)), desc=f'rendering virtual view for {image_name}'):
        if i == 0:
            render_rgb = src_rgb.cpu().numpy()
            wrap_rgb = src_rgb.cpu().numpy()
            mask = np.ones_like(src_rgb.cpu().numpy())
            wrap_rgb_path = os.path.join(save_dir, f'{i:04d}_condition.png')
            render_rgb_path = os.path.join(save_dir, f'{i:04d}.png')
            mask_path = os.path.join(save_dir, f'{i:04d}_mask.png')
            cv2.imwrite(wrap_rgb_path, cv2.cvtColor((wrap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(render_rgb_path, cv2.cvtColor((render_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            continue

        # shift and rotate
        shift_direction = lane_shift_directions(ego_frame_poses, src_camera.meta['frame'])
        shift_direction = torch.from_numpy(shift_direction).float().to('cuda', non_blocking=True)
        ego_pose = copy.deepcopy(src_camera.ego_pose)
        ego_pose[:3, 3] += shift_direction * shift * r
        c, s = math.cos(yaw * r), math.sin(yaw * r)
        rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).float().to('cuda', non_blocking=True)
        ego_pose[:3, :3] = rot @ ego_pose[:3, :3]

        # set camera pose
        ext = src_camera.extrinsic
        tar_extrinsic = ego_pose @ ext
        tar_intrinsic = copy.deepcopy(src_intrinsic)
        tar_camera = copy.deepcopy(src_camera)
        tar_camera.set_extrinsic(tar_extrinsic)
        tar_camera.set_intrinsic(tar_intrinsic)
        tar_extrinsics.append(tar_extrinsic)
        tar_intrinsics.append(tar_intrinsic)

        with torch.no_grad():
            tar_result = renderer.render(tar_camera, model)
            tar_depth = process_depth(tar_result['depth'], tar_result['acc'])
            tar_depth = tar_depth.permute(1, 2, 0)
            tar_rgb = tar_result['rgb'].permute(1, 2, 0)
            tar_rgbs.append(tar_rgb)
            tar_depths.append(tar_depth)

    tar_rgbs = torch.stack(tar_rgbs, dim=0)
    tar_depths = torch.stack(tar_depths, dim=0)
    tar_extrinsics = torch.stack(tar_extrinsics, dim=0)
    tar_intrinsics = torch.stack(tar_intrinsics, dim=0)
    b = tar_depths.shape[0]
    src_rgbs = src_rgb.unsqueeze(0).repeat(b, 1, 1, 1)
    src_depths = src_depth.unsqueeze(0).repeat(b, 1, 1, 1)
    src_intrinsics = src_intrinsic.unsqueeze(0).repeat(b, 1, 1)
    src_extrinsics = src_extrinsic.unsqueeze(0).repeat(b, 1, 1)

    wrap_rgbs, masks = virtual_wrap_kernel_torch(
        tar_intrinsics,
        tar_extrinsics,
        tar_depths,
        tar_rgbs,
        src_intrinsics,
        src_extrinsics,
        src_depths,
        src_rgbs,
    )
    # b, h, w, 3   b, h, w
    print(f'saving virtual view for {image_name}')
    for i in tqdm(range(b)):
        render_rgb = tar_rgbs[i].cpu().numpy()
        wrap_rgb = wrap_rgbs[i].cpu().numpy()
        mask = masks[i].cpu().numpy()
        render_rgb_path = os.path.join(save_dir, f'{i+1:04d}.png')
        wrap_rgb_path = os.path.join(save_dir, f'{i+1:04d}_condition.png')
        mask_path = os.path.join(save_dir, f'{i+1:04d}_mask.png')
        cv2.imwrite(wrap_rgb_path, cv2.cvtColor((wrap_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(render_rgb_path, cv2.cvtColor((render_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
