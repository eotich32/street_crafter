import sys
import os
import numpy as np
import cv2
import argparse
import imageio
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.pcd_utils import fetchPly
from utils.multiprocess_utils import track_parallel_progress
from utils.render_utils import render_pointcloud_pytorch3d, render_pointcloud_diff_point_rasterization
from data_processor.waymo_processor.waymo_helpers import load_ego_poses, load_calibration, load_track, get_lane_shift_direction, image_filename_to_frame, image_heights, image_widths, LANE_SHIFT_SIGN
from easyvolcap.utils.console_utils import *

class WaymoLiDARRenderer(object):
    def __init__(self, args, scene_ids):
        self.data_dir = args.data_dir
        self.skip_existing = args.skip_existing
        self.delta_frames = args.delta_frames
        self.cams = args.cams
        self.scene_ids = scene_ids
        self.gpus = args.gpus
        self.save_dir = args.save_dir
        self.workers = args.workers
        self.shifts = args.shifts

        self.flip_axis = 1

        if 'train' in self.data_dir:
            self.split = 'train'
        else:
            self.split = 'val'

    def read_lidar_ply(self, lidar_dir, ego_poses, trajectory):
        ply_dict = dict()

        # read background ply
        lidar_background_dir = os.path.join(lidar_dir, 'background')
        ply_dict_background = dict()

        bkgd_ply_list = sorted([os.path.join(lidar_background_dir, f) for f in os.listdir(lidar_background_dir) if f.endswith('.ply') and f != 'full.ply'])
        for i, bkgd_ply_path in enumerate(tqdm(bkgd_ply_list, desc='Reading background ply')):
            bkgd_ply = fetchPly(bkgd_ply_path)

            mask = bkgd_ply.mask
            xyz_vehicle = bkgd_ply.points[mask]
            xyz_vehicle_homo = np.concatenate([xyz_vehicle, np.ones_like(xyz_vehicle[..., :1])], axis=-1)
            frame = image_filename_to_frame(os.path.basename(bkgd_ply_path))
            xyz_world = xyz_vehicle_homo @ ego_poses[frame].T
            xyz_world = xyz_world[..., :3]
            rgb = bkgd_ply.colors[mask]
            ply_dict_background[frame] = np.concatenate([xyz_world, rgb], axis=-1)

        ply_dict['background'] = ply_dict_background

        # read actor lidar
        lidar_actor_dir = os.path.join(lidar_dir, 'actor')
        for track_id in os.listdir(lidar_actor_dir):
            ply_dict_actor = dict()

            lidar_actor_dir_ = os.path.join(lidar_actor_dir, track_id)
            actor_ply_list = sorted([os.path.join(lidar_actor_dir_, f) for f in os.listdir(lidar_actor_dir_)
                                     if f.endswith('.ply') and f != 'full.ply'])
            for actor_ply_path in actor_ply_list:
                frame = image_filename_to_frame(os.path.basename(actor_ply_path))
                actor_ply = fetchPly(actor_ply_path)
                mask = actor_ply.mask

                if mask.sum() == 0:
                    continue

                xyz = actor_ply.points[mask]
                rgb = actor_ply.colors[mask]
                rigid = not trajectory[track_id]['deformable']
                if rigid and self.split == 'val':
                    num_pointcloud_1 = (xyz[:, self.flip_axis] > 0).sum()
                    num_pointcloud_2 = (xyz[:, self.flip_axis] < 0).sum()
                    if num_pointcloud_1 >= num_pointcloud_2:
                        xyz_part = xyz[xyz[:, self.flip_axis] > 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] > 0]
                    else:
                        xyz_part = xyz[xyz[:, self.flip_axis] < 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] < 0]
                    xyz_flip = xyz_part.copy()
                    xyz_flip[:, self.flip_axis] *= -1
                    rgb_flip = rgb_part.copy()
                    xyz = np.concatenate([xyz, xyz_flip], axis=0)
                    rgb = np.concatenate([rgb, rgb_flip], axis=0)

                ply_dict_actor[frame] = np.concatenate([xyz, rgb], axis=-1)

            ply_dict[track_id] = ply_dict_actor

        return ply_dict

    def read_images(self, scene_dir, cam, num_frames):
        images = []
        for frame in tqdm(range(num_frames), desc='Reading images'):
            image_path = os.path.join(scene_dir, 'images', f'{frame:06d}_{cam}.png')
            image = cv2.imread(image_path)[..., [2, 1, 0]] / 255.
            images.append(image)
        return images

    def make_lidar_ply(self, ply_dict, start_frame, end_frame):
        ply_frame_dict = dict()
        bkgd_ply = []
        for frame in range(start_frame, end_frame + 1):
            bkgd_ply.append(ply_dict['background'][frame])
        bkgd_ply = np.concatenate(bkgd_ply, axis=0)  # [N, xyz + rgb]
        ply_frame_dict['background'] = bkgd_ply

        for track_id in ply_dict.keys():
            if track_id == 'background':
                continue
            actor_ply = []
            for frame in range(start_frame, end_frame + 1):
                if frame not in ply_dict[track_id]:
                    continue
                actor_ply.append(ply_dict[track_id][frame])

            # empty actor ply
            if len(actor_ply) == 0:
                continue

            actor_ply = np.concatenate(actor_ply, axis=0)

            ply_frame_dict[track_id] = actor_ply
        return ply_frame_dict

    def transform_lidar_ply(self, lidar_ply, pose):
        xyz, rgb = lidar_ply[..., :3], lidar_ply[..., 3:]
        xyz_homo = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz = (xyz_homo @ pose.T)[..., :3]
        lidar_ply = np.concatenate([xyz, rgb], axis=-1)
        return lidar_ply

    def check_existing(self, scene_dir, cam, save_dir, num_frames):
        for frame in range(num_frames):
            image_proj_save_path = os.path.join(scene_dir, 'lidar', save_dir, f'{str(frame).zfill(6)}_{cam}.png')
            mask_proj_save_path = os.path.join(scene_dir, 'lidar', save_dir, f'{str(frame).zfill(6)}_{cam}_mask.png')
            if not (os.path.exists(image_proj_save_path) and os.path.exists(mask_proj_save_path)):
                return False
        return True

    def render(self):
        """Convert action."""
        print("Start rendering ...")
        id_list = self.scene_ids
        render_fn = self.render_one

        chunk = 32
        for i in range(0, len(id_list), chunk):
            print(f"\nProcessing scenes {i:03d} to {min(i+chunk-1, len(id_list)-1):03d} ...")
            cur_id_list = id_list[i:i + chunk]
            track_parallel_progress(render_fn, cur_id_list, self.workers)
        print("\nFinished ...")

    def get_save_dir(self, scene_dir, shift):
        save_dir = os.path.join(scene_dir, 'lidar', self.save_dir) if shift == 0 \
                else os.path.join(scene_dir, 'lidar', f'{self.save_dir}_shift_{shift:.2f}')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def render_one(self, scene_idx):
        """Project action for single file."""
        scene_dir = os.path.join(self.data_dir, str(scene_idx).zfill(3))

        # load camera poses
        ego_frame_poses, ego_cam_poses = load_ego_poses(datadir=scene_dir)

        # load calibration
        extrinsics, intrinsics = load_calibration(datadir=scene_dir)

        # load tracks
        track_info, _, trajectory = load_track(scene_dir)

        lidar_dir = os.path.join(scene_dir, 'lidar')
        lidar_render_dir = os.path.join(lidar_dir, self.save_dir)

        os.makedirs(lidar_render_dir, exist_ok=True)
        num_frames = len(ego_frame_poses)
        
        # gpu_id
        gpu_id = self.gpus[scene_idx % len(self.gpus)]
        print('Using GPU:', gpu_id)
        
        for cam in self.cams:
            print(f'Processing scene {scene_idx:03d}, camera {cam}')
            ply_dict = self.read_lidar_ply(lidar_dir=lidar_dir, ego_poses=ego_frame_poses, trajectory=trajectory)

            for shift in self.shifts:
                ply_render_result_rgb = []
                render_save_dir = self.get_save_dir(scene_dir, shift)
                if self.skip_existing and self.check_existing(scene_dir, cam, render_save_dir, num_frames):
                    print(f'Skipping scene {scene_idx:03d}, camera {cam}, shift {shift}')
                    continue

                print(f'Processing scene {scene_idx:03d}, camera {cam}, shift {shift}')
                for frame in tqdm(range(num_frames), desc=f'Rendering'):
                    start_frame = max(0, frame - self.delta_frames)
                    end_frame = min(num_frames - 1, frame + self.delta_frames)
                    track_info_frame = track_info[f'{frame:06d}']
                    ego_pose = ego_cam_poses[cam, frame]

                    # step1: get point cloud xyz and rgb
                    ply_frame_dict = self.make_lidar_ply(ply_dict, start_frame=start_frame, end_frame=end_frame)                    
                    ply_frame = [ply_frame_dict.pop('background')]
                    for track_id, ply_actor_frame in ply_frame_dict.items():
                        if track_id not in track_info_frame:
                            continue

                        if shift == 0:
                            #camera_box = track_info_frame[track_id]['camera_box']
                            #lidar_box = track_info_frame[track_id]['lidar_box']
                            box = track_info_frame[track_id]
                        else:
                            box = track_info_frame[track_id]['lidar_box']

                        pose_vehicle = np.eye(4)
                        pose_vehicle[:3, :3] = np.array([
                            [np.cos(box['heading']), -np.sin(box['heading']), 0],
                            [np.sin(box['heading']), np.cos(box['heading']), 0],
                            [0, 0, 1]
                        ])
                        pose_vehicle[:3, 3] = np.array([box['center_x'], box['center_y'], box['center_z']])
                        pose_vehicle = ego_pose @ pose_vehicle
                        ply_actor_frame = self.transform_lidar_ply(ply_actor_frame, pose_vehicle)
                        ply_frame.append(ply_actor_frame)
                    ply_frame = np.concatenate(ply_frame, axis=0)
                    ply_xyz, ply_rgb = ply_frame[..., :3], ply_frame[..., 3:]

                    # step2: modify camera pose
                    ego_pose_shift = ego_pose.copy()
                    lane_shift_direction = get_lane_shift_direction(ego_frame_poses, frame)
                    lane_shift_sign = LANE_SHIFT_SIGN[f'{scene_idx:03d}']

                    ego_pose_shift[:3, 3] += lane_shift_sign * lane_shift_direction * shift
                    c2w = ego_pose_shift @ extrinsics[cam]
                    w2c = np.linalg.inv(c2w)
                    ixt = intrinsics[cam]
                    h, w = image_heights[cam], image_widths[cam]

                    # step3: filter out invisible points
                    ply_xyz_cam = np.dot(ply_xyz, w2c[:3, :3].T) + w2c[:3, 3:].T
                    ply_depth = ply_xyz_cam[:, 2]
                    ply_pixel = np.dot(ply_xyz_cam, ixt.T)
                    ply_pixel = ply_pixel[:, :2] / ply_pixel[:, 2:]
                    ply_valid = (
                        (ply_depth > 1e-3)
                        & (ply_pixel[:, 0] >= 0)
                        & (ply_pixel[:, 0] < w)
                        & (ply_pixel[:, 1] >= 0)
                        & (ply_pixel[:, 1] < h),
                    )[0]
                    ply_pixel = ply_pixel[ply_valid]
                    ply_pixel = np.round(ply_pixel).astype(np.int32)
                    valid_ply_depth = ply_depth[ply_valid]
                    valid_ply_rgb = ply_rgb[ply_valid]
                    valid_ply_mask = np.ones_like(valid_ply_depth)
                    valid_ply_feature = np.concatenate([valid_ply_rgb, valid_ply_depth[:, None], valid_ply_mask[:, None]], axis=-1)
                    valid_ply_xyz = ply_xyz[ply_valid]

                    # step4: render point cloud
                    ply_render = render_pointcloud_diff_point_rasterization(c2w, ixt, valid_ply_xyz, valid_ply_feature, h, w, gpu_id, use_ndc_scale=True, scale=0.01)
                    # ply_render = render_pointcloud_pytorch3d(c2w, ixt, valid_ply_xyz, valid_ply_feature, h, w, gpu_id)
                    ply_render_rgb, ply_render_mask = ply_render[0, ..., :3], ply_render[0, ..., 3]
                    ply_render_rgb = ply_render_rgb.detach().cpu().numpy()
                    ply_render_mask = ply_render_mask.detach().cpu().numpy()

                    image_render_save_path = os.path.join(render_save_dir, f'{str(frame).zfill(6)}_{cam}.png')
                    mask_render_save_path = os.path.join(render_save_dir, f'{str(frame).zfill(6)}_{cam}_mask.png')

                    image_render = (ply_render_rgb * 255).astype(np.uint8)
                    cv2.imwrite(image_render_save_path, image_render[..., [2, 1, 0]])

                    mask_render = (ply_render_mask * 255).astype(np.uint8)
                    cv2.imwrite(mask_render_save_path, mask_render)

                    ply_render_result_rgb.append(ply_render_rgb)

                image_video_save_path = os.path.join(render_save_dir, f'render_rgb_{cam}.mp4')
                imageio.mimwrite(image_video_save_path, [(x * 255).astype(np.uint8) for x in ply_render_result_rgb], fps=10)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./waymo_processor/training_set_processed')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing files')
    parser.add_argument('--delta_frames', type=int, default=10)
    parser.add_argument('--cams', type=int, nargs='+', default=[0])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='color_render')
    parser.add_argument('--shifts', type=int, nargs='+', default=[0])

    args = parser.parse_args()
    data_dir = args.data_dir
    scene_ids = sorted([x for x in os.listdir(data_dir)])
    scene_ids = [int(x) for x in scene_ids]
    dataset_projector = WaymoLiDARRenderer(args, scene_ids)
    if args.workers == 1:
        for scene_idx in scene_ids:
            dataset_projector.render_one(scene_idx)
    else:
        dataset_projector.render()


if __name__ == '__main__':
    main()
