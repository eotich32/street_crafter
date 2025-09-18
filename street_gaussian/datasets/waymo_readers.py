from easyvolcap.utils.console_utils import *
from street_gaussian.utils.waymo_utils import generate_dataparser_outputs
from street_gaussian.utils.graphics_utils import focal2fov
from street_gaussian.utils.data_utils import get_val_frames
from street_gaussian.datasets.base_readers import CameraInfo, SceneInfo, getNerfppNorm
from street_gaussian.utils.novel_view_utils import waymo_novel_view_cameras, append_interpolated_novel_view
from data_processor.waymo_processor.waymo_helpers import image_filename_to_frame, image_filename_to_cam
from street_gaussian.config import cfg
import copy
from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
sys.path.append(os.getcwd())


def set_reference_frames(cam_infos):
    """
    为cam_infos列表中每个is_original=False的元素设置ref_frame（即在原始轨迹中插值生成的新帧），
    指向列表中其他is_original=True元素中帧编号最接近的那一个的image_name

    参数:
        cam_infos: 包含元素的列表，每个元素有:
                   - metadata属性，包含'is_original'和'frame'的字典
                   - image_name属性，表示该帧的图像名称
    """
    # 首先收集所有is_original为True的元素及其帧号和image_name
    interpolated_frames = [
        info
        for info in cam_infos
        if not info.metadata.get('is_original', False)
    ]
    original_frames = cam_infos

    # 如果原始帧数量小于2，没有可参考的帧，直接返回
    if len(interpolated_frames) < 2:
        return

    # 为每个原始帧找到最接近的其他原始帧的image_name
    for info in interpolated_frames:
        frame = info.metadata['frame']#, info.image_path
        min_diff = None
        closest_image_name = None

        # 遍历其他原始帧，找到最接近的帧对应的image_name
        for other_info in original_frames:
            # 跳过自身
            if info is other_info:
                continue
            other_frame, other_image_name = other_info.metadata['frame'], other_info.image_path

            # 计算帧差的绝对值
            diff = abs(frame - other_frame)

            # 更新最小差值和对应的image_name
            if min_diff is None or diff < min_diff and other_info.metadata.get('is_original', False):
                min_diff = diff
                closest_image_name = other_image_name

        # 设置参考帧的image_name到ref_frame
        if closest_image_name is not None:
            info.metadata['ref_frame_image'] = closest_image_name


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

    load_interpolated = cfg.data.get('load_interpolated', False)
    interpolated_frame_mapping = None
    if load_interpolated:
        interpolated_mapping_file = os.path.join(path, 'interpolated_track', 'frame_mapping.json')
        if os.path.exists(interpolated_mapping_file):
            with open(interpolated_mapping_file, "r") as f:
                interpolated_frame_mapping = json.load(f)

    # dynamic mask
    if load_interpolated:
        dynamic_mask_dir = os.path.join(path, 'interpolated_dynamic_mask')
    else:
        dynamic_mask_dir = os.path.join(path, 'dynamic_mask')
    load_dynamic_mask = True

    # sky mask
    if load_interpolated:
        sky_mask_dir = os.path.join(path, 'interpolated_sky_mask')
    else:
        sky_mask_dir = os.path.join(path, 'sky_mask')
    load_sky_mask = (cfg.mode == 'train')

    # lidar depth
    if load_interpolated:
        lidar_depth_dir = os.path.join(path, 'interpolated_lidar/depth')
    else:
        lidar_depth_dir = os.path.join(path, 'lidar/depth')
    load_lidar_depth = (cfg.mode == 'train')

    output = generate_dataparser_outputs(
        datadir=path,
        selected_frames=selected_frames,
        cameras=cfg.data.get('cameras', [0, 1, 2]),
        load_interpolated=load_interpolated
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
        img_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path).split('.')[0]
        if interpolated_frame_mapping is not None and not interpolated_frame_mapping[str(frames_idx[i])]['is_original']:
            # 插值之后，生成的新view没有对应的image（作为新视角用diffusion补全），此时不能直接读image，因为这个imagepath只是空文件占位符
            # 在后面调用append_interpolated_novel_view函数里面，会生成雷达图填充这个image，见append_interpolated_novel_view
            image = None
            cam = image_filename_to_cam(image_name)
            frame = image_filename_to_frame(image_name)
            while image is None:
                try:
                    image = Image.open(os.path.join(img_dir, f'{frame:06d}_{cam}.{cfg.data.get("img_format", "png")}'))
                except Exception as e:
                    frame -= 1
            # width, height = output['img_resolutions'][cam]['width'], output['img_resolutions'][cam]['height']
            width, height = image.size
            is_original = False
        else:
            image = Image.open(image_path)
            width, height = image.size
            is_original = True
        fx, fy = ixt[0, 0], ixt[1, 1]

        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)

        c2w = ego_pose @ ext
        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()

        metadata = dict()
        metadata['width'] = width
        metadata['height'] = height
        metadata['is_original'] = is_original
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = ego_pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]
        metadata['is_novel_view'] = False
        guidance_dir = os.path.join(cfg.source_path, 'lidar', f'color_render')
        metadata['guidance_rgb_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}.{cfg.data.get("img_format", "png")}')
        metadata['guidance_mask_path'] = os.path.join(guidance_dir, f'{str(frames[i]).zfill(6)}_{cams[i]}_mask.{cfg.data.get("img_format", "png")}')

        guidance = dict()

        # load dynamic mask
        if load_dynamic_mask and (interpolated_frame_mapping is None or interpolated_frame_mapping[str(frames_idx[i])]['is_original']):
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.{cfg.data.get("img_format", "png")}')
            obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
            guidance['obj_bound'] = Image.fromarray(obj_bound)

        # load lidar depth
        if load_lidar_depth and (interpolated_frame_mapping is None or interpolated_frame_mapping[str(frames_idx[i])]['is_original']):
            depth_path = os.path.join(lidar_depth_dir, f'{image_name}.npz')
            depth = np.load(depth_path)
            mask = depth['mask'].astype(np.bool_)
            value = depth['value'].astype(np.float32)
            depth = np.zeros_like(mask).astype(np.float32)
            depth[mask] = value
            guidance['lidar_depth'] = depth

        # load sky mask
        if load_sky_mask and (interpolated_frame_mapping is None or interpolated_frame_mapping[str(frames_idx[i])]['is_original']):
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.{cfg.data.get("img_format", "png")}')
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
    original_cameras_for_novel_view = [cam_info for cam_info in cam_infos if 'is_original' in cam_info.metadata and cam_info.metadata['is_original']] # 插值的不在这里生成新视角，只有原始视角才需要调用waymo_novel_view_cameras生成新视角
    novel_view_cam_infos = waymo_novel_view_cameras(original_cameras_for_novel_view, ego_frame_poses, obj_info, cams_tracklets)

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

    # 对于插值生成的视角，设置其最临近的原始视角的image_name为其参考的图像，用于one step diffusion之后的difix后处理。（difix后处理需要一个参考图，此时用最临近的视角）
    set_reference_frames(cam_infos)

    if interpolated_frame_mapping is not None and cfg.mode == 'train':
        scene_metadata['interpolated_frame_mapping'] = interpolated_frame_mapping
        novel_cameras = [cam_info for cam_info in cam_infos if not interpolated_frame_mapping[str(cam_info.metadata['frame_idx'])]['is_original']]
        append_interpolated_novel_view(novel_view_cam_infos, novel_cameras, obj_info, cams_tracklets)
        train_cam_infos = [cam_info for cam_info in cam_infos if interpolated_frame_mapping[str(cam_info.metadata['frame_idx'])]['is_original']]
        test_cam_infos = [cam_info for cam_info in cam_infos if interpolated_frame_mapping[str(cam_info.metadata['frame_idx'])]['is_original']]

    print(f'Scene extent: {nerf_normalization["radius"]}')

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        metadata=scene_metadata,
        novel_view_cameras=novel_view_cam_infos,
    )

    return scene_info
