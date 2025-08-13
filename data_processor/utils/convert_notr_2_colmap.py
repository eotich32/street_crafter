from tqdm import tqdm
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import random
import cv2


def matrix_to_euler_xyz(matrix):
    rotation_matrix = matrix[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('zyx', False)
    translation_vector = matrix[:3, 3]
    return euler_angles, translation_vector


def calculate_w2c(ego_pose, extrinsic):
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    # extrinsic = extrinsic @ OPENCV2DATASET
    c2w = ego_pose @ extrinsic
    # rot, pos = matrix_to_euler_xyz(c2w)
    rot, pos = c2w[:3,:3], c2w[:3,3]
    # rotationInverse = np.linalg.inv(Rotation.from_euler('zyx', rot, False).as_matrix())
    rotationInverse = np.linalg.inv(rot)
    pos = -rotationInverse.dot(pos)
    q = Rotation.from_matrix(rotationInverse).as_quat()
    w, x, y, z = q[3], q[0], q[1], q[2]
    return w,x,y,z,pos


if __name__ == '__main__':
    camera_names = ['left_rear','left_front','front_main','right_front','right_rear', 'rear_main']
    input_notr_dir = r'D:\Projects\3dgs_datas\dataset\qirui\notr'
    output_dir = r'D:\Projects\3dgs_datas\dataset\qirui\notr/colmap'
    colmap_modle_path = os.path.join(output_dir, 'sparse/0')
    if not os.path.exists(colmap_modle_path):
        os.makedirs(colmap_modle_path)

    output_img_path = os.path.join(output_dir, 'images')
    for cam_name in camera_names:
        if not os.path.exists(os.path.join(output_img_path, cam_name)):
            os.makedirs(os.path.join(output_img_path, cam_name))

    ego_pose_dir = os.path.join(input_notr_dir, 'ego_pose')
    extrinsics_dir = os.path.join(input_notr_dir, 'extrinsics')
    intrinsics_dir = os.path.join(input_notr_dir, 'intrinsics')
    lidar_dir = os.path.join(input_notr_dir, 'lidar')

    # convert extrinsic
    extrinsics = {}
    for filename in os.listdir(extrinsics_dir):
        camera_id = filename.split('.')[0]
        # if int(camera_id) > 2:
        #     continue
        extrinsic = np.loadtxt(os.path.join(extrinsics_dir, filename))
        extrinsics[camera_id] = extrinsic

    # first_frame_pos = None
    widths = {}
    heights = {}
    with open(os.path.join(colmap_modle_path, "images.txt"), 'w') as f:
        i = 1
        for img_filename in tqdm(os.listdir(os.path.join(input_notr_dir, 'images')), desc="Processing images"):
            frame_cameraId = img_filename.split('.')[0].split('_')
            frame = frame_cameraId[0]
            camera_id = frame_cameraId[1]
            ego_pose = np.loadtxt(os.path.join(ego_pose_dir, frame + '.txt')) 
            # if first_frame_pos is None:
            #     first_frame_pos = ego_pose[:3,3].copy()
            # ego_pose[:3,3] -= first_frame_pos
            if camera_id in extrinsics:
                w,x,y,z,pos = calculate_w2c(ego_pose, extrinsics[camera_id])
                img_name = camera_names[int(camera_id)] + '/' + frame + '_' + camera_id + '.jpg'
                source_img_path = os.path.join(input_notr_dir, 'images', frame + '_' + camera_id + '.jpg')
                dest_img_path = os.path.join(output_img_path, camera_names[int(camera_id)], frame + '_' + camera_id + '.jpg')
                if camera_id not in heights or camera_id not in widths:
                    image = cv2.imread(source_img_path)
                    heights[camera_id], widths[camera_id] = image.shape[0], image.shape[1]
                shutil.copyfile(source_img_path, dest_img_path)
                f.write('{} {} {} {} {} {} {} {} {} {} \n\n'.format(i,w,x,y,z,pos[0], pos[1], pos[2],camera_id, img_name))
                i = i+1

    # convert intrinsic
    with open(os.path.join(colmap_modle_path, "cameras.txt"), 'w') as f:
        for filename in os.listdir(intrinsics_dir):
            camera_id = filename.split('.')[0]
            intrinsic = np.loadtxt(os.path.join(intrinsics_dir, filename))
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            output = '{} {} {} {} {} {} {} {}'.format(camera_id, 'PINHOLE', widths[camera_id], heights[camera_id], fx, fy, cx, cy)
            f.write(output + '\n')

    # convert lidar points
    # Note that in the Waymo Open Dataset, the lidar coordinate system is the same
    # as the vehicle coordinate system
    lidar_to_worlds = {}
    # we tranform the poses w.r.t. the first timestep to make the origin of the
    # first ego pose as the origin of the world coordinate system.
    # ego_to_world_start = np.loadtxt(
    #     os.path.join(os.path.join(ego_pose_dir, '000.txt'))
    # )
    # for filename in os.listdir(ego_pose_dir):
    #     frame = filename.split('.')[0]
    #     ego_pose = np.loadtxt(os.path.join(ego_pose_dir, filename))
    #     # compute ego_to_world transformation
    #     lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_pose
    #     lidar_to_worlds[frame] = lidar_to_world

    # origins, directions, ranges, laser_ids = [], [], [], []
    # # flow/ground info are used for evaluation only
    # flows, flow_classes, grounds = [], [], []
    # # in waymo, we simplify timestamps as the time indices
    # timestamps, timesteps = [], []

    # all_points = []
    # with open(os.path.join(colmap_modle_path, "points3D.txt"), 'w') as output:
    #     i = 0
    #     for filename in os.listdir(lidar_dir):
    #         frame = filename.split('.')[0]
    #         # each lidar_info contains an Nx14 array
    #         # from left to right:
    #         # origins: 3d, points: 3d, flows: 3d, flow_class: 1d,
    #         # ground_labels: 1d, intensities: 1d, elongations: 1d, laser_ids: 1d
    #         lidar_info = np.memmap(os.path.join(lidar_dir, filename),dtype=np.float32,mode="r",).reshape(-1, 14)
    #         original_length = len(lidar_info)

    #         # laser_ids: 0: TOP, 1: FRONT, 2: SIDE_LEFT, 3: SIDE_RIGHT, 4: REAR
    #         lidar_info = lidar_info[lidar_info[:, 13] == 0]
    #         lidar_points = lidar_info[:, 3:6]

    #         # transform lidar points from lidar coordinate system to world coordinate system
    #         lidar_points = (
    #                 lidar_to_worlds[frame][:3, :3] @ lidar_points.T
    #                 + lidar_to_worlds[frame][:3, 3:4]
    #         ).T
    #         num_points = lidar_points.shape[0]
    #         random_indices = np.random.choice(num_points, size=int(num_points * 0.0001), replace=False)
    #         random_points = lidar_points[random_indices]
    #         for p in random_points:
    #             r = random.randint(0, 255)
    #             g = random.randint(0, 255)
    #             b = random.randint(0, 255)
    #             # rgb后面的几个值是fake的，3DGS中没用。这里为了让colmap gui加载时不崩溃
    #             output.write(" ".join([str(c) for c in [i + 1, p[0],p[1],p[2], r, g, b, 0.01, 1, 0, 2, 0, 3,0]]) + "\n")
    #             i = i+1
