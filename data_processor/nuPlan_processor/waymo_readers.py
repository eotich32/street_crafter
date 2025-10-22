from easyvolcap.utils.console_utils import *
from street_gaussian.utils.waymo_utils import generate_dataparser_outputs
from street_gaussian.utils.graphics_utils import focal2fov
from street_gaussian.utils.data_utils import get_val_frames
from street_gaussian.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm
from street_gaussian.utils.novel_view_utils import waymo_novel_view_cameras
from street_gaussian.config import cfg
from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
sys.path.append(os.getcwd())


def readWaymoInfo(path, images='images', split_train=-1, split_test=-1, **kwargs):
    selected_frames = cfg.data.get('selected_frames', None)
    if cfg.debug:
        selected_frames = [0, 0]

    if cfg.data.get('load_pcd_from', False) and (cfg.mode == 'train'):
        load_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'input_ply')
        save_dir = os.path.join(cfg.model_path, 'input_ply')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(load_dir, save_dir)

        colmap_dir = os.path.join(cfg.workspace, cfg.data.load_pcd_from, 'colmap')
        save_dir = os.path.join(cfg.model_path, 'colmap')
        os.system(f'rm -rf {save_dir}')
        shutil.copytree(colmap_dir, save_dir)

    # dynamic mask
    dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = True

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train')

    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar/depth')
    load_lidar_depth = (cfg.mode == 'train')

    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=selected_frames,
        cameras=cfg.data.get('cameras', [0, 1, 2, 3, 4, 5, 6, 7]),
    )

    exts = output['exts']
    ixts = output['ixts']
    ego_cam_poses = output['ego_cam_poses']
    ego_frame_poses = output['ego_frame_poses']
    image_filenames = output['image_filenames']
    obj_info = output['obj_info']
    frames, cams, frames_idx = output['frames'], output['cams'], output['frames_idx']
    cams_timestamps = output['cams_timestamps']
    cams_tracklets = output['cams_tracklets']

    num_frames = output['num_frames']
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['camera_tracklets'] = cams_tracklets
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = len(cfg.data.cameras)
    scene_metadata['num_frames'] = num_frames
    scene_metadata['ego_frame_poses'] = ego_frame_poses
    scene_metadata['camera_timestamps'] = dict()
    for cam_idx in cfg.data.get('cameras'):
        scene_metadata['camera_timestamps'][cam_idx] = sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if cams[i] == cam_idx])
        # scene_metadata['camera_timestamps'][cam_idx]['train_timestamps'] = \
        #     sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if frames_idx[i] in train_frames and cams[i] == cam_idx])
        # scene_metadata['camera_timestamps'][cam_idx]['test_timestamps'] = \
        #     sorted([timestamp for i, timestamp in enumerate(cams_timestamps) if frames_idx[i] in test_frames and cams[i] == cam_idx])

    # make camera infos: train, test, novel view cameras
    cam_infos = []
    for i in tqdm(range(len(exts)), desc='Preparing cameras and images'):
        # prepare camera pose and image
        ext = exts[i]
        ixt = ixts[i]
        ego_pose = ego_cam_poses[i]
        image_path = image_filenames[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        c2w = ego_pose @ ext
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()

        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = ego_pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]
        metadata['is_novel_view'] = False
        guidance_dir = os.path.join(cfg.source_path, 'lidar', f'color_render')
        metadata['guidance_rgb_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}.png')
        metadata['guidance_mask_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}_mask.png')

        guidance = dict()

        # load dynamic mask
        if load_dynamic_mask:
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
            obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
            guidance['obj_bound'] = Image.fromarray(obj_bound)

        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npz')
            depth = np.load(depth_path)
            mask = depth['mask'].astype(np.bool_)
            value = depth['value'].astype(np.float32)
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
            guidance['lidar_depth'] = depth

        # load sky mask
        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            guidance['sky_mask'] = Image.fromarray(sky_mask)

        mask = None
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            metadata=metadata,
            guidance=guidance,
        )
        cam_infos.append(cam_info)

    train_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['frame_idx'] in train_frames]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['frame_idx'] in test_frames]
    for cam_info in train_cam_infos:
        cam_info.metadata['is_val'] = False
    for cam_info in test_cam_infos:
        cam_info.metadata['is_val'] = True

    print('making novel view cameras')
    novel_view_cam_infos = waymo_novel_view_cameras(cam_infos, ego_frame_poses, obj_info, cams_tracklets)

    # 3
    # Get scene extent
    # 1. Default nerf++ setting
    if cfg.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(novel_view_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    # 2. The radius we obtain should not be too small (larger than 10 here)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)

    # 3. If we have extent set in config, we ignore previous setting
    if cfg.data.get('extent', False):
        nerf_normalization['radius'] = cfg.data.extent

    # 4. We write scene radius back to config
    cfg.data.extent = float(nerf_normalization['radius'])

    # 5. We write scene center and radius to scene metadata
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )

    return scene_info
