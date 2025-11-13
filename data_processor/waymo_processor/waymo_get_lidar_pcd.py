import os
import numpy as np
import cv2
import math
import argparse
import sys
# import open3d as o3d
from PIL import Image
from tqdm import tqdm

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_cartesian
from waymo_open_dataset.v2.perception import (
    box as _v2_box,
    camera_image as _v2_camera_image,
    context as _v2_context,
    lidar as _v2_lidar,
    pose as _v2_pose,
)

sys.path.append(os.getcwd())
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_helpers import load_calibration, load_track, get_object, ParquetReader, load_ego_poses
from utils.img_utils import visualize_depth_numpy
from utils.box_utils import bbox_to_corner3d, inbbox_points
from utils.pcd_utils import storePly, fetchPly
from utils.base_utils import project_numpy

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

from typing import Dict, List, Optional, Tuple, TypedDict
import tensorflow as tf
DUMMY_DISTANCE_VALUE = 2e3  # meters, used for missing points
np.set_printoptions(precision=4, suppress=True)
import transforms3d
from copy import deepcopy

def convert_range_image_to_point_cloud(
    range_image: _v2_lidar.RangeImage,
    calibration: _v2_context.LiDARCalibrationComponent,
    pixel_pose: Optional[_v2_lidar.PoseRangeImage] = None,
    frame_pose: Optional[_v2_pose.VehiclePoseComponent] = None,
    keep_polar_features=False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Converts one range image from polar coordinates to point cloud.
        same as in wod api, but return the mask in addition plus channel id

    Args:
        range_image: One range image return captured by a LiDAR sensor.
        calibration: Parameters for calibration of a LiDAR sensor.
        pixel_pose: If not none, it sets pose for each range image pixel.
        frame_pose: This must be set when `pose` is set.
        keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.

    Returns:
        A 3 [N, D] tensor of 3D LiDAR points. D will be 3 if keep_polar_features is
        False (x, y, z) and 6 if keep_polar_features is True (range, intensity,
        elongation, x, y, z).
            1. Lidar points-cloud
            2. Missing points points-cloud
            3. Range image mask above dummy distance.

    """

    # missing points are found directly from range image
    val_clone = deepcopy(range_image.tensor.numpy())  # type: ignore
    no_return = val_clone[..., 0] == -1  # where range is -1
    val_clone[..., 0][no_return] = DUMMY_DISTANCE_VALUE
    # re-assign the field
    object.__setattr__(range_image, "values", val_clone.flatten())

    if pixel_pose is not None:
        assert frame_pose is not None
        # From range image, missing points do not have a pose.
        # So we replace their pose with the vehicle pose.
        # pixel pose & frame pose
        pixel_pose_clone = deepcopy(pixel_pose.tensor.numpy())  # type: ignore
        pixel_pose_mask = pixel_pose_clone[..., 0] == 0
        tr_orig = frame_pose.world_from_vehicle.transform.reshape(4, 4)  # type: ignore
        rot = tr_orig[:3, :3]
        x, y, z = tr_orig[:3, 3]
        yaw, pitch, roll = transforms3d.euler.mat2euler(rot, "szyx")
        # ` [roll, pitch, yaw, x, y, z]`
        pixel_pose_clone[..., 0][pixel_pose_mask] = roll
        pixel_pose_clone[..., 1][pixel_pose_mask] = pitch
        pixel_pose_clone[..., 2][pixel_pose_mask] = yaw
        pixel_pose_clone[..., 3][pixel_pose_mask] = x
        pixel_pose_clone[..., 4][pixel_pose_mask] = y
        pixel_pose_clone[..., 5][pixel_pose_mask] = z
        # re-assign the field
        object.__setattr__(pixel_pose, "values", pixel_pose_clone.flatten())

    range_image_cartesian = convert_range_image_to_cartesian(
        range_image=range_image,
        calibration=calibration,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        keep_polar_features=keep_polar_features,
    )

    range_image_tensor = range_image.tensor
    range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]  # 0  # type: ignore
    points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))
    missing_points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(~range_image_mask))

    return points_tensor, missing_points_tensor, range_image_mask

def extract_pointwise_camera_projection(
    range_image: _v2_lidar.RangeImage,
    camera_projection: _v2_lidar.CameraProjectionRangeImage,
) -> tf.Tensor:
  """Extracts information about where in camera images each point is projected.

  Args:
    range_image: One range image return captured by a LiDAR sensor.
    camera_projection: LiDAR point to camera image projections.

  Returns:
    A [N, 6] tensor of camera projection per point. See
      lidar.CameraProjectionRangeImage for definitions of inner dimensions.
  """
  range_image_tensor = range_image.tensor
  range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]  # 0  # type: ignore
  camera_project_tensor = camera_projection.tensor
  pointwise_camera_projection_tensor = tf.gather_nd(camera_project_tensor, tf.compat.v1.where(range_image_mask))
  missing_points_camera_projection_tensor = tf.gather_nd(camera_project_tensor, tf.compat.v1.where(~range_image_mask))

  return pointwise_camera_projection_tensor, missing_points_camera_projection_tensor, range_image_mask


def save_lidar(root_dir, seq_path, seq_save_dir):
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')

    base_seq_name = os.path.basename(seq_path).split('.')[0]
    seq_context = base_seq_name[8:-19]
    seq_reader = ParquetReader(context_name=seq_context, dataset_dir=root_dir)

    lidar_calib = seq_reader("lidar_calibration").compute()
    lidar_proj_df = seq_reader("lidar_camera_projection").compute()
    lidar_df = seq_reader("lidar").compute()
    lidar_pose_df = seq_reader("lidar_pose").compute()
    vehicle_pose_df = seq_reader("vehicle_pose").compute()

    os.makedirs(seq_save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_dir = os.path.join(seq_save_dir, 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)
    lidar_dir_background = os.path.join(lidar_dir, 'background')
    os.makedirs(lidar_dir_background, exist_ok=True)
    lidar_dir_actor = os.path.join(lidar_dir, 'actor')
    os.makedirs(lidar_dir_actor, exist_ok=True)
    lidar_dir_depth = os.path.join(lidar_dir, 'depth')
    os.makedirs(lidar_dir_depth, exist_ok=True)
    

    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(lidar_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []
    
    print("Processing LiDAR data...")

    for frame_id, (_, v) in tqdm(enumerate(vehicle_pose_df.iterrows())):
        xyzs = []
        camera_projections = []
        missing_xyzs = []
        VehiclePoseCom = v2.VehiclePoseComponent.from_dict(v)
        lidar_df_frame = lidar_df[lidar_df["key.frame_timestamp_micros"] == VehiclePoseCom.key.frame_timestamp_micros]
        
        for _, r in lidar_df_frame.iterrows():
            LidarComp = v2.LiDARComponent.from_dict(r)
            
            lidar_pose_df_ = lidar_pose_df[
                (lidar_pose_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_pose_df["key.laser_name"] ==  LidarComp.key.laser_name)
            ]
            if len(lidar_pose_df_) == 0:
                pixel_pose = None
                frame_pose = None
            else:    
                LidarPoseComp = v2.LiDARPoseComponent.from_dict(lidar_pose_df_.iloc[0])
                pixel_pose = LidarPoseComp.range_image_return1
                frame_pose = VehiclePoseCom

            lidar_proj_df_ = lidar_proj_df[
                (lidar_proj_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_proj_df["key.laser_name"] == LidarComp.key.laser_name)
            ]
            LidarProjComp = v2.LiDARCameraProjectionComponent.from_dict(lidar_proj_df_.iloc[0])
            
            lidar_calib_ = lidar_calib[lidar_calib["key.laser_name"] == LidarComp.key.laser_name]
            LidarCalibComp = v2.LiDARCalibrationComponent.from_dict(lidar_calib_.iloc[0])
            
            pts_lidar, missing_pts, _ = convert_range_image_to_point_cloud(
                LidarComp.range_image_return1,
                LidarCalibComp,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose,
                keep_polar_features=False,
            )
            missing_pts = missing_pts.numpy()

            pts_projection, missing_projection, _ = extract_pointwise_camera_projection(
                LidarComp.range_image_return1,
                LidarProjComp.range_image_return1,
            )
            
            xyzs.append(pts_lidar.numpy())
            camera_projections.append(pts_projection.numpy())
            missing_xyzs.append(missing_pts)

        xyzs = np.concatenate(xyzs, axis=0)
        camera_projections = np.concatenate(camera_projections, axis=0)
        missing_xyzs = np.concatenate(missing_xyzs, axis=0)

        rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.uint8)        
        camera_id = camera_projections[:, 0]
        masks = camera_id > 0
        
        # Generate lidar depth and get pointcloud rgb
        for i in range(5):
            image_filename = os.path.join(image_dir, f'{frame_id:06d}_{i}.png')
            image = cv2.imread(image_filename)[..., [2, 1, 0]].astype(np.uint8)      
            h, w = image.shape[:2]
            
            depth_filename = os.path.join(lidar_dir_depth, f'{frame_id:06d}_{i}.npz')
            depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
            depth_vis_filename = os.path.join(lidar_dir_depth, f'{frame_id:06d}_{i}.png')
            
            # Sprase lidar depth
            num_pts = xyzs.shape[0]
            pts_idx = np.arange(num_pts)
            pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
            pts_camera_id = camera_projections.reshape(-1, 3)[:, 0] 
            mask_depth_idx = (pts_camera_id == i+1)
            mask_depth = pts_idx[mask_depth_idx]
            
            xyzs_mask = xyzs[mask_depth]
            xyzs_mask_homo = np.concatenate([xyzs_mask, np.ones_like(xyzs_mask[..., :1])], axis=-1)
            
            c2w = extrinsics[i]
            w2c = np.linalg.inv(c2w)
            xyzs_mask_cam = xyzs_mask_homo @ w2c.T
            xyzs_mask_depth = xyzs_mask_cam[..., 2]
            xyzs_mask_depth = np.clip(xyzs_mask_depth, a_min=1e-1, a_max=1e2)
            
            u_depth, v_depth = camera_projections[mask_depth, 1], camera_projections[mask_depth, 2]
            u_depth = np.clip(u_depth, 0, w-1).astype(np.int32)
            v_depth = np.clip(v_depth, 0, h-1).astype(np.int32)
            indices = v_depth * w + u_depth
            
            np.minimum.at(depth, indices, xyzs_mask_depth)
            depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
            valid_depth_pixel = (depth != 0)
            valid_depth_value = depth[valid_depth_pixel].astype(np.float32)
            valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
            
            np.savez_compressed(depth_filename, mask=valid_depth_pixel, value=valid_depth_value)
            
            # missing_uv, mask = project_numpy(
            #     xyz=missing_xyzs, 
            #     K=intrinsics[i], 
            #     RT=np.linalg.inv(c2w), 
            #     H=h, W=w
            # )
            # missing_u = np.clip(missing_uv[mask, 0], 0, w-1).astype(np.int32)
            # missing_v = np.clip(missing_uv[mask, 1], 0, h-1).astype(np.int32)
            # missing_mask = np.zeros_like(valid_depth_pixel, dtype=np.bool_)
            # missing_mask[missing_v, missing_u] = True
                        
            try:
                if i == 0:
                    depth = depth.reshape(h, w).astype(np.float32)
                    depth_vis, _ = visualize_depth_numpy(depth)
                    depth_on_img = image[..., [2, 1, 0]]
                    depth_on_img[depth > 0] = depth_vis[depth > 0]
                    cv2.imwrite(depth_vis_filename, depth_on_img)      
            except:
                print(f'error in visualize depth of {image_filename}, depth range: {depth.min()} - {depth.max()}')
            
            # Colorize 
            mask_rgb = (camera_id == i+1)
            if mask_rgb.sum() != 0:
                # use the first projected camera
                u_rgb, v_rgb = camera_projections[mask_rgb, 1], camera_projections[mask_rgb, 2]
                u_rgb = np.clip(u_rgb, 0, w-1).astype(np.int32)
                v_rgb = np.clip(v_rgb, 0, h-1).astype(np.int32)
                rgb = image[v_rgb, u_rgb]
                rgbs[mask_rgb] = rgb
            
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f'{frame_id:06d}']
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue
            
            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(frame_id)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
            
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            
            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = rgbs[inbbox_mask]
            masks_inbbox = masks[inbbox_mask]
            
            pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
            pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
            pointcloud_actor[track_id]['mask'].append(masks_inbbox)
            
            masks_inbbox = masks_inbbox[..., None]
            ply_actor_path = os.path.join(lidar_dir_actor, track_id, f'{frame_id:06d}.ply')
            try:
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, masks_inbbox)
            except:
                pass # No pcd

        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        masks_background = masks_background[..., None]
        ply_background_path = os.path.join(lidar_dir_background, f'{frame_id:06d}.ply')
        
        storePly(ply_background_path, xyzs_background, rgbs_background, masks_background)
    
    for track_id, pointcloud in pointcloud_actor.items():
        xyzs = np.concatenate(pointcloud['xyz'], axis=0)
        rgbs = np.concatenate(pointcloud['rgb'], axis=0)
        masks = np.concatenate(pointcloud['mask'], axis=0)
        masks = masks[..., None]
        ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, 'full.ply')
        
        try:
            storePly(ply_actor_path_full, xyzs, rgbs, masks)
        except:
            pass # No pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='./test_data/')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir

    i = 90
    sequence_path = '17612470202990834368_2800_000_2820_000'
    print(f'{i}: {sequence_path}')
    sequence_save_dir = os.path.join(save_dir, str(i).zfill(3))

    save_lidar(
        root_dir=root_dir,
        seq_path=sequence_path,
        seq_save_dir=sequence_save_dir,
    )

    
if __name__ == '__main__':
    main()