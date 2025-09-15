import io
import os
import numpy as np
import cv2
import imageio
import sys
import math
import argparse
import json
import pickle
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from google.protobuf.json_format import MessageToDict

sys.path.append(os.getcwd())
from waymo_open_dataset import label_pb2
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" '
        ">1.4.5 to install the official devkit first."
    )
    
from waymo_helpers import load_track, load_calibration, load_ego_poses, get_object, \
    project_label_to_image, project_label_to_mask, draw_3d_box_on_img, opencv2camera


camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT', 
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}
    
def parse_seq_rawdata(process_list, seq_path, seq_save_dir, skip_existing):
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')

    os.makedirs(seq_save_dir, exist_ok=True)
        
    # set start and end timestep
    datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
    num_frames = sum(1 for _ in datafile)
    start_idx = 0
    end_idx = num_frames - 1
    
    # ego pose 
    ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose')
    if 'pose' not in process_list:
        print("Skipping pose processing...")
    elif os.path.exists(ego_pose_save_dir) and skip_existing:
        print('Ego pose already exists, skipping...')
    else:
        os.makedirs(ego_pose_save_dir, exist_ok=True)
        print("Processing ego pose...")
        timestamp = dict()
        timestamp['FRAME'] = dict()
        for camera_name in camera_names_dict.values():
            timestamp[camera_name] = dict()
        
        datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
        for frame_id, data in tqdm(enumerate(datafile)):
            frame = dataset_pb2.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))
            if len(frame.map_features) > 0:
                print(f'Frame {frame_id} has {len(frame.map_features)} map features, dumping to map_features.json')
                with open(os.path.join(seq_save_dir, f"map_features_{frame_id}.json"), 'w') as f:
                    json_obj = [MessageToDict(x, preserving_proto_field_name=True) for x in frame.map_features]
                    json.dump(json_obj, f, indent=1)
            pose = np.array(frame.pose.transform).reshape(4, 4)
            np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}.txt"), pose)
            timestamp['FRAME'][str(frame_id).zfill(6)] = frame.timestamp_micros / 1e6        
        
            camera_calibrations = frame.context.camera_calibrations
            for i, camera in enumerate(camera_calibrations):
                camera_name = camera.name
                camera_name_str = camera_names_dict[camera_name]
                camera = get_object(frame.images, camera_name)
                camera_timestamp = camera.pose_timestamp
                timestamp[camera_name_str][str(frame_id).zfill(6)] = camera_timestamp
                
                camera_pose = np.array(camera.pose.transform).reshape(4, 4)
                np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}_{camera_name-1}.txt"), camera_pose)
        
        timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
        with open(timestamp_save_path, 'w') as f:
            json.dump(timestamp, f, indent=1)
    
    # camera calibration 
    intrinsic_save_dir = os.path.join(seq_save_dir, 'intrinsics')
    extrinsic_save_dir = os.path.join(seq_save_dir, 'extrinsics')

    if 'calib' not in process_list:
        print("Skipping calibration processing...")
    elif os.path.exists(intrinsic_save_dir) and os.path.exists(extrinsic_save_dir) and skip_existing:
        print('Calibration already exists, skipping...')
    else:
        os.makedirs(intrinsic_save_dir, exist_ok=True)
        os.makedirs(extrinsic_save_dir, exist_ok=True)
        print("Processing camera calibration...")
        
        datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
        for frame_id, data in tqdm(enumerate(datafile)):
            frame = dataset_pb2.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))
            camera_calibrations = frame.context.camera_calibrations
            break
        
        extrinsics = []
        intrinsics = []
        camera_names = []
        for camera in camera_calibrations:
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            extrinsic = np.matmul(extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            camera_names.append(camera.name)
        
        for i in range(5):
            np.savetxt(os.path.join(extrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), extrinsics[i])
            np.savetxt(os.path.join(intrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), intrinsics[i])
    
    # image
    image_save_dir = os.path.join(seq_save_dir, 'images')
    if 'image' not in process_list:
        print("Skipping image processing...")
    elif os.path.exists(image_save_dir) and skip_existing:
        print('Images already exist, skipping...')
    else:
        os.makedirs(image_save_dir, exist_ok=True)      
        print("Processing image data...")
        
        camera_timestamp = dict()
        datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
        
        for frame_id, data in tqdm(enumerate(datafile)):
            frame = dataset_pb2.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))          
            for img in frame.images:
                img_path = os.path.join(image_save_dir, f'{frame_id:06d}_{str(img.name - 1)}.png')
                with open(img_path, "wb") as fp:
                    fp.write(img.image)

            # for camera_name, camera_name_str in camera_names_dict.items():       
            #     camera = [obj for obj in frame.images if obj.name == camera_name][0]
            #     img = np.array(Image.open(io.BytesIO(camera.image))) 
            #     img_path = os.path.join(image_save_dir, f'{frame_id:06d}_{str(camera.name - 1)}.png')
            #     cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                camera_timestamp[f'{frame_id:06d}_{str(img.name - 1)}'] = img.pose_timestamp
                
        camera_timestamp_save_path = os.path.join(seq_save_dir, 'images', 'timestamps.json')
        with open(camera_timestamp_save_path, 'w') as f:
            json.dump(camera_timestamp, f, indent=1)
        
        print("Processing image data done...")

    # trajectory
    track_dir = os.path.join(seq_save_dir, "track")
    if 'track' not in process_list:
        print("Skipping tracking data processing...")
    elif os.path.exists(track_dir) and skip_existing:
        print('Tracking data already exists, skipping...')
    else:
        os.makedirs(track_dir, exist_ok=True)
        print("Processing tracking data...")

        track_info = dict() # 以每个frame的一个bbox为单位 frame_id, track_id 记录LiDAR-synced和Camera_synced bboxes
        track_camera_visible = dict() # 以每个camera的一个bbox为单位 frame_id, camera_id, track_id 记录这个camera看到了哪些物体
        trajectory_info = dict() # 以每个track物体的一个bbox为单位 track_id, frame_id 记录LiDAR-synced boxes
    
        object_ids = dict() # 每个物体的track_id对应一个数字 （track_id, object_id）之后streetgaussian训练时用的是object_id

        track_vis_imgs = []

        datafile = tf.data.TFRecordDataset(seq_path, compression_type="")

        for frame_id, data in tqdm(enumerate(datafile)):
            frame = dataset_pb2.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))  
        
            track_info_cur_frame = dict()
            track_camera_visible_cur_frame = dict()

            timestamp = frame.timestamp_micros / 1e6

            images = dict()
            for camera_name in camera_names_dict.keys():
                camera = [obj for obj in frame.images if obj.name == camera_name][0]
                img = np.array(Image.open(io.BytesIO(camera.image))) 
                images[camera_name] = img
                
                track_camera_visible_cur_frame[camera_name-1] = []
            
            for label in frame.laser_labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    obj_class = "vehicle"
                elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                    obj_class = "pedestrian"
                elif label.type == label_pb2.Label.Type.TYPE_SIGN:
                    obj_class = "sign"
                elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
                    obj_class = "cyclist"
                else:
                    obj_class = "misc"
                
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])      
            
                label_id = label.id
                
                # Add one label
                if label_id not in trajectory_info.keys():
                    trajectory_info[label_id] = dict()
                    
                if label.id not in object_ids:
                    object_ids[label.id] = len(object_ids)
                
                track_info_cur_frame[label_id] = dict()
                
                # LiDAR-synced box
                lidar_synced_box = dict()
                lidar_synced_box['height'] = label.box.height
                lidar_synced_box['width'] = label.box.width
                lidar_synced_box['length'] = label.box.length
                lidar_synced_box['center_x'] = label.box.center_x
                lidar_synced_box['center_y'] = label.box.center_y
                lidar_synced_box['center_z'] = label.box.center_z
                lidar_synced_box['heading'] = label.box.heading
                lidar_synced_box['label'] = obj_class
                lidar_synced_box['speed'] = speed
                lidar_synced_box['timestamp'] = timestamp
                track_info_cur_frame[label_id]['lidar_box'] = lidar_synced_box                
                trajectory_info[label_id][f'{frame_id:06d}'] = lidar_synced_box
                
                # Camera-synced box
                if True:#label.camera_synced_box.ByteSize(): TODO
                    camera_synced_box = dict()
                    camera_synced_box['height'] = label.box.height
                    camera_synced_box['width'] = label.box.width
                    camera_synced_box['length'] = label.box.length
                    camera_synced_box['center_x'] = label.box.center_x
                    camera_synced_box['center_y'] = label.box.center_y
                    camera_synced_box['center_z'] = label.box.center_z
                    camera_synced_box['heading'] = label.box.heading
                    camera_synced_box['label'] = obj_class
                    camera_synced_box['speed'] = speed
                    track_info_cur_frame[label_id]['camera_box'] = camera_synced_box
                    
                    c = math.cos(camera_synced_box['heading'])
                    s = math.sin(camera_synced_box['heading'])
                    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = rotz_matrix
                    obj_pose_vehicle[:3, 3] = np.array([camera_synced_box['center_x'], camera_synced_box['center_y'], camera_synced_box['center_z']])
                    
                    camera_visible = []
                    for camera_name in camera_names_dict.keys():
                        camera_calibration = [obj for obj in frame.context.camera_calibrations if obj.name == camera_name][0]
                        vertices, valid = project_label_to_image(
                            dim=[camera_synced_box['length'], camera_synced_box['width'], camera_synced_box['height']],
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                    
                        # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                        # partial visible for the case when not all corners can be observed
                        if valid.any():
                            camera_visible.append(camera_name-1)
                            track_camera_visible_cur_frame[camera_name-1].append(label_id)
                        if valid.all() and camera_name in [dataset_pb2.CameraName.FRONT_LEFT, dataset_pb2.CameraName.FRONT, dataset_pb2.CameraName.FRONT_RIGHT]:
                            vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                            draw_3d_box_on_img(vertices, images[camera_name])
                    
                    # print(f'At frame {frame_id}, label {label_id} has a camera-synced box visible from cameras {camera_visible}')
                else:
                    track_info_cur_frame[label.id]['camera_box'] = None
                    
            track_info[f'{frame_id:06d}'] = track_info_cur_frame
            track_camera_visible[f'{frame_id:06d}'] = track_camera_visible_cur_frame
                
            track_vis_img = np.concatenate([
                images[dataset_pb2.CameraName.FRONT_LEFT], 
                images[dataset_pb2.CameraName.FRONT], 
                images[dataset_pb2.CameraName.FRONT_RIGHT]], axis=1)
            track_vis_imgs.append(track_vis_img)

        ego_frame_poses, _ = load_ego_poses(seq_save_dir, False)

        # reset information for trajectory 
        # poses, stationary, symmetric, deformable
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
            
            # if label_id == '-ItvfksmEcYtVEcOjjRESg':
            #     __import__('ipdb').set_trace()
            
            dims = np.array(dims).astype(np.float32)
            dim = np.max(dims, axis=0)
            poses_vehicle = np.array(poses_vehicle).astype(np.float32)
            poses_world = np.array(poses_world).astype(np.float32)
            actor_world_postions = poses_world[:, :3, 3]
            
            # if label == 'sign':
            #     __import__('ipdb').set_trace()
            
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
            
            # print(new_trajectory['label'], new_trajectory['stationary'])
        
        # save visualization        
        imageio.mimwrite(os.path.join(track_dir, "track_vis.mp4"), track_vis_imgs, fps=24)
        
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

        print("Processing tracking data done...")
    
    
    
    lidar_dir = os.path.join(seq_save_dir, 'lidar')
    if 'lidar' not in process_list:
        print("Skipping LiDAR processing...")
    elif os.path.exists(lidar_dir) and skip_existing:
        print('LiDAR already exists, skipping...')
    else:
        os.makedirs(lidar_dir, exist_ok=True)
        print("Processing LiDAR data...")
        lidar_dir_background = os.path.join(lidar_dir, 'background')
        os.makedirs(lidar_dir_background, exist_ok=True)
        lidar_dir_actor = os.path.join(lidar_dir, 'actor')
        os.makedirs(lidar_dir_actor, exist_ok=True)
        lidar_dir_depth = os.path.join(lidar_dir, 'depth')
        os.makedirs(lidar_dir_depth, exist_ok=True)
        
        track_info, track_camera_visible, trajectory = load_track(seq_save_dir, load_interpolated=False)
        extrinsics, intrinsics = load_calibration(seq_save_dir)

        pointcloud_actor = dict()
        for label_id, traj in trajectory.items():
            dynamic = not traj['stationary']
            if dynamic and traj['label'] != 'sign':
                os.makedirs(os.path.join(lidar_dir_actor, label_id), exist_ok=True)
                pointcloud_actor[label_id] = dict()
                pointcloud_actor[label_id]['xyz'] = []
                pointcloud_actor[label_id]['rgb'] = []
                pointcloud_actor[label_id]['mask'] = []

        datafile = tf.data.TFRecordDataset(seq_path, compression_type="")
        for frame_id, data in tqdm(enumerate(datafile)):
            frame = dataset_pb2.Frame() 
            frame.ParseFromString(bytearray(data.numpy()))
            # too slow .....
            # range_images, camera_projections, seg_labels, range_image_top_pose = parse_range_image_and_camera_projection(frame)
        
    dynamic_mask_dir = os.path.join(seq_save_dir, 'dynamic_mask')
    if 'dynamic' not in process_list:
        print("Skipping dynamic mask processing...")
    elif os.path.exists(dynamic_mask_dir) and skip_existing:
        print('Dynamic mask already exists, skipping...')
    else:
        os.makedirs(dynamic_mask_dir, exist_ok=True)
        print("Processing dynamic mask...")
        track_info, track_camera_visible, trajectory = load_track(seq_save_dir, load_interpolated=False)
        extrinsics, intrinsics = load_calibration(seq_save_dir)
        for frame, track_info_frame in track_info.items():
            track_camera_visible_cur_frame = track_camera_visible[frame]
            for cam, track_ids in track_camera_visible_cur_frame.items():
                dynamic_mask_name = f'{frame}_{cam}.png'
                dynamic_mask = np.zeros((image_heights[cam], image_widths[cam]), dtype=np.uint8).astype(np.bool_)
                
                deformable_mask_name = f'{frame}_{cam}_deformable.png'
                deformable_mask = np.zeros((image_heights[cam], image_widths[cam]), dtype=np.uint8).astype(np.bool_)
                
                calibration_dict = dict()
                calibration_dict['extrinsic'] = extrinsics[cam]
                calibration_dict['intrinsic'] = intrinsics[cam]
                calibration_dict['height'] = image_heights[cam]
                calibration_dict['width'] = image_widths[cam]      
                
                for track_id in track_ids:
                    object_tracklet = trajectory[track_id]
                    if object_tracklet['stationary']:
                        continue
                    pose_idx = trajectory[track_id]['frames'].index(int(frame))
                    pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]
                    height, width, length = trajectory[track_id]['height'], trajectory[track_id]['width'], trajectory[track_id]['length']
                    box_mask = project_label_to_mask(
                        dim=[length, width, height],
                        obj_pose=pose_vehicle,
                        calibration=None,
                        calibration_dict=calibration_dict,
                    )

                    dynamic_mask = np.logical_or(dynamic_mask, box_mask)
                    if trajectory[track_id]['deformable']:
                        deformable_mask = np.logical_or(deformable_mask, box_mask)
                
                dynamic_mask_path = os.path.join(dynamic_mask_dir, dynamic_mask_name)
                cv2.imwrite(dynamic_mask_path, dynamic_mask.astype(np.uint8) * 255)
        
                # deformable_mask_path = os.path.join(dynamic_mask_dir, deformable_mask_name)
                # cv2.imwrite(deformable_mask_path, deformable_mask.astype(np.uint8) * 255)
                
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'track', 'lidar', 'dynamic'])
    # parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'track'])
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='./test_data/')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    
    all_sequence_names = sorted([x for x in os.listdir(root_dir) if x.endswith('.tfrecord')])
    all_sequence_paths = [os.path.join(root_dir, x) for x in all_sequence_names]
    for i, sequence_path in enumerate(all_sequence_paths):
        print(f'{i}: {sequence_path}')
        sequence_save_dir = os.path.join(save_dir, str(i).zfill(3))
        parse_seq_rawdata(
            process_list=process_list,
            seq_path=sequence_path,
            seq_save_dir=sequence_save_dir,
            skip_existing=args.skip_existing,
        )
        
if __name__ == '__main__':
    main()
    
