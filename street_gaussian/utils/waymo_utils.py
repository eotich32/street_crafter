import os
import numpy as np
import cv2
import torch
import json
import math
from bidict import bidict
from glob import glob
from tqdm import tqdm
from street_gaussian.config import cfg
from street_gaussian.utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from data_processor.waymo_processor.waymo_helpers import image_heights, image_widths, image_filename_to_frame, image_filename_to_cam
from data_processor.waymo_processor.waymo_helpers import _camera2label, _label2camera, waymo_track2label, load_camera_info, load_track

from easyvolcap.utils.console_utils import *
from PIL import Image
# box_info: box_center_x box_center_y box_center_z box_heading


# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world


def get_obj_pose_tracking(datadir, selected_frames, cameras):
    track_info, track_camera_visible, trajectory = load_track(datadir)
    object_ids_path = os.path.join(datadir, 'track/track_ids.json')
    with open(object_ids_path, 'r') as f:
        object_ids = json.load(f)
        object_ids = bidict(object_ids)

    start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1
    visible_track_ids = []
    for frame in range(start_frame, end_frame + 1):
        frame_camera_visible = track_camera_visible[f'{frame:06d}']
        for cam in cameras:
            visible_id = frame_camera_visible[cam]
            visible_track_ids += visible_id
    unique_track_ids = sorted(list(set(visible_track_ids)))  # consistency is important here
    unique_track_ids = [track_id for track_id in unique_track_ids if not trajectory[track_id]['stationary']]

    box_scale = cfg.data.get('box_scale', 1.0)
    print('box scale: ', box_scale)

    objects_info = dict()
    for i, track_id in enumerate(unique_track_ids):
        object_info = dict()
        object_info['id'] = i  # 0, 1, 2, ...
        object_id = object_ids[track_id]
        object_info['object_id'] = object_id  # 0, 1, 2, ...
        object_info['track_id'] = track_id  # 'kJb_yYfoGxVj5fJdi0Sn5A' ....
        object_info['class'] = trajectory[track_id]['label']
        object_info['class_label'] = waymo_track2label[object_info['class']]
        object_info['height'] = trajectory[track_id]['height']
        object_info['width'] = trajectory[track_id]['width'] * box_scale
        object_info['length'] = trajectory[track_id]['length'] * box_scale
        object_info['deformable'] = trajectory[track_id]['deformable']

        frames = trajectory[track_id]['frames']
        object_info['start_frame'] = min(frames)
        object_info['end_frame'] = max(frames)
        objects_info[object_id] = object_info

    if len(objects_info) > 0:
        objects_tracklets_vehicle = np.ones([num_frames, len(objects_info), 5]) * -1.0
        for frame_idx, frame in enumerate(list(range(start_frame, end_frame + 1))):
            for object_id, object_info in objects_info.items():
                start_frame, end_frame = object_info['start_frame'], object_info['end_frame']
                if not (start_frame <= frame and end_frame >= frame):
                    continue
                id = object_info['id']
                track_id = object_info['track_id']
                track_info_frame = track_info[f'{frame:06d}']
                track_info_obj = track_info_frame[track_id]['lidar_box']
                box_info = [track_info_obj['center_x'], track_info_obj['center_y'], track_info_obj['center_z'], track_info_obj['heading'], 1]
                objects_tracklets_vehicle[frame_idx, id] = np.array(box_info)
    else:
        print("No moving actors in current sequence")
        objects_tracklets_vehicle = np.ones([num_frames, 1, 5]) * -1.0

    return objects_tracklets_vehicle, objects_info


def generate_dataparser_outputs(
    datadir,
    selected_frames=None,
    cameras=[0, 1, 2, 3, 4],
    load_interpolated=True,
):
    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir, load_interpolated)

    if load_interpolated:
        image_dir = os.path.join(datadir, 'interpolated_images')
    else:
        image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.png')))
    num_frames_all = len(ego_frame_poses)

    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1


    # load camera, frame, path
    frames, frames_idx, cams, image_filenames = [], [], [], []
    ixts, exts, poses, c2ws = [], [], [], []
    cams_timestamps = []
    tracklet_timestamps = []

    if load_interpolated:
        timestamp_path = os.path.join(datadir, 'interpolated_track', 'timestamps.json')
    else:
        timestamp_path = os.path.join(datadir, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)

    for frame in range(start_frame, end_frame + 1):
        tracklet_timestamps.append(timestamps[_label2camera[0]][f'{frame:06d}'])  # assume the tracklet timestamp is the same as the front camera

    img_resolutions = {}
    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if cam not in img_resolutions:
            image = Image.open(image_filename)
            width, height = image.size
            img_resolutions[cam] = {'width': width, 'height': height}

        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)

            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)

            camera_name = _label2camera[cam]
            timestamp = timestamps[camera_name][f'{frame:06d}']
            cams_timestamps.append(timestamp)

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + tracklet_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    tracklet_timestamps = np.array(tracklet_timestamps) - timestamp_offset

    object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        datadir,
        selected_frames,
        cameras,
    )

    # object_tracklets_vehicle: [num_frames, max_obj_per_frame, [x, y, z, heading, valid]]
    # camera_tracklets: [num_cams, num_frames, max_obj_per_frame, [x, y, z, qx, qy, qz, qw, valid]]

    if len(object_info) == 0:
        camera_tracklets = np.ones([len(cameras), num_frames, 1, 8]) * -1
    else:
        object_idx_all = dict()
        for object_id in object_info.keys():
            id = object_info[object_id]['id']
            object_valid = object_tracklets_vehicle[:, id, -1]  # (num_frames,)
            object_idx_all[object_id] = np.argwhere(object_valid == 1).astype(np.int32)[:, 0]

        # get camera tracklets
        camera_tracklets = np.ones([len(cameras), num_frames, len(object_info), 8]) * -1

        for i in range(len(cams)):
            cam = cams[i]
            frame_idx = frames_idx[i]
            frame = frames[i]
            timestamp = cams_timestamps[i]
            ego_pose = ego_frame_poses[frame]
            for k, v in object_info.items():
                object_id = v['object_id']
                id = v['id']
                start_frame, end_frame = v['start_frame'], v['end_frame']
                if not (start_frame <= frame and end_frame >= frame):
                    continue

                object_idx = object_idx_all[object_id]
                if object_idx.shape[0] == 1:
                    tracklet = object_tracklets_vehicle[object_idx[0], id]
                    pose = tracklet[:4]
                else:
                    timestamps = tracklet_timestamps[object_idx]  # type: ignore
                    delta_timestamps = np.abs(timestamps - timestamp)
                    idx1, idx2 = np.argsort(delta_timestamps)[:2]
                    tracklet1 = object_tracklets_vehicle[object_idx[idx1], id]
                    tracklet2 = object_tracklets_vehicle[object_idx[idx2], id]
                    timestamp1, timestamp2 = tracklet_timestamps[object_idx[idx1]], tracklet_timestamps[object_idx[idx2]]

                    pose1 = tracklet1[:4]
                    pose2 = tracklet2[:4]

                    if np.all(pose1 == -1) or np.all(pose2 == -1):
                        breakpoint()

                    alpha = (timestamp - timestamp2) / (timestamp1 - timestamp2)
                    pose = alpha * pose1 + (1 - alpha) * pose2

                pose_vehicle, pose_world = make_obj_pose(ego_pose, pose)

                camera_tracklets[cam, frame_idx, id, :7] = pose_world
                camera_tracklets[cam, frame_idx, id, 7] = 1

    result = dict()

    # image
    result['image_filenames'] = image_filenames

    # camera pose
    result['exts'] = exts
    result['ixts'] = ixts
    result['c2ws'] = c2ws
    result['ego_cam_poses'] = poses
    result['ego_frame_poses'] = ego_frame_poses

    # actor pose
    result['cams_tracklets'] = camera_tracklets
    result['obj_info'] = object_info

    # camera index, frame index and timestamp
    result['num_frames'] = num_frames
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['cams_timestamps'] = cams_timestamps
    result['img_resolutions'] = img_resolutions

    # run colmap
    colmap_basedir = os.path.join(f'{cfg.model_path}/colmap')
    if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
        from script.colmap_waymo_full import run_colmap_waymo
        run_colmap_waymo(result)

    return result
