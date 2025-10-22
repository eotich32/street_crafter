import glob
import numpy as np
import os
import pickle
import cv2
from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Type
def image_filename_to_cam(x): return int(x.split('.')[0][-1])
def image_filename_to_frame(x): return int(x.split('.')[0][:6])


image_heights = [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280]
image_widths = [1920, 1920, 1920, 1920, 1920, 1920, 1920, 1920]

_camera2label = {
    'CAM_L0': 0,
    'CAM_F0': 1,
    'CAM_R0': 2,
    'CAM_R1': 3,
    'CAM_R2': 4,
    'CAM_B0': 5,
    'CAM_L2': 6,
    'CAM_L1': 7,

}

_label2camera = {
    0:  'CAM_L0',
    1:  'CAM_F0',
    2:  'CAM_R0',
    3:  'CAM_R1',
    4:  'CAM_R2',
    5:  'CAM_B0',
    6:  'CAM_L2',
    7:  'CAM_L1',
}

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}
LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
    "173": 1,
    "176": 1,
    "159": -1,
    "140": -1,
    "121": -1,
    "101": 1,
    "096": -1,
    "090": -1,
    "079": -1,
    "067": 1, 
    "062": -1,
    "051": -1,
    "049": -1,
    "035": -1,
    "027": -1,
    "020": -1,
    }
)

class ParquetReader:
    def __init__(self, context_name: str, dataset_dir: str = "/data/dataset/wod/training", nb_partitions: int = 120):
        self.context_name = context_name
        self.dataset_dir = dataset_dir
        self.nb_partitions = nb_partitions

    def read(self, tag: str):
        import dask.dataframe as dd
        """Creates a Dask DataFrame for the component specified by its tag."""
        paths = glob.glob(f"{self.dataset_dir}/{tag}/{self.context_name}.parquet")
        print("判断path值")
        print(f"{self.dataset_dir}/{tag}/{self.context_name}.parquet")
        return dd.read_parquet(paths, npartitions=self.nb_partitions)  # type: ignore

    def __call__(self, tag: str):
        return self.read(tag)


def get_object(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]


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

# load ego pose and camera calibration(extrinsic and intrinsic)


def load_ego_poses(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(8)]
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


def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    assert os.path.exists(extrinsics_dir), f"{extrinsics_dir} does not exist"
    intrinsics_dir = os.path.join(datadir, 'intrinsics')
    assert os.path.exists(intrinsics_dir), f"{intrinsics_dir} does not exist"

    intrinsics = []
    extrinsics = []
    for i in range(8):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))

    for i in range(8):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    return extrinsics, intrinsics


# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(8):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(8):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(8)]
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

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(8)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


opencv2camera = np.array([[0., 0., 1., 0.],
                          [-1., 0., 0., 0.],
                          [0., -1., 0., 0.],
                          [0., 0., 0., 1.]])


def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)  # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera)  # [forward, left, up] to [right, down, forward]
    return extrinsic


def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic


def project_label_to_image(dim, obj_pose, calibration):
    from utils.base_utils import project_numpy
    from utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3],
        K=intrinsic,
        RT=np.linalg.inv(extrinsic),
        H=height, W=width
    )
    return points_uv, valid


def project_label_to_mask(dim, obj_pose, calibration, calibration_dict=None):
    from utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame

    if calibration_dict is not None:
        extrinsic = calibration_dict['extrinsic']
        intrinsic = calibration_dict['intrinsic']
        width = calibration_dict['width']
        height = calibration_dict['height']
    else:
        extrinsic = get_extrinsic(calibration)
        intrinsic = get_intrinsic(calibration)
        width, height = calibration.width, calibration.height

    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic),
        H=height, W=width
    )

    return mask
'''
def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)

    # Draw a cross on the front face to identify front & back.
    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)
'''
def draw_3d_box_on_img(vertices, img, color=(0, 255, 0), thickness=2):
    """
    vertices: (8, 2) array or list of 8 2D points
    img: H×W×3 BGR image, in-place drawing
    """
    vertices = np.asarray(vertices, dtype=int).reshape(8, 2)
    # 12 条边
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        cv2.line(img, tuple(vertices[i]), tuple(vertices[j]), color, thickness)

def get_lane_shift_direction(ego_frame_poses, frame):
    assert frame >= 0 and frame < len(ego_frame_poses)
    if frame == 0:
        ego_pose_delta = ego_frame_poses[1][:3, 3] - ego_frame_poses[0][:3, 3]
    else:
        ego_pose_delta = ego_frame_poses[frame][:3, 3] - ego_frame_poses[frame - 1][:3, 3]

    ego_pose_delta = ego_pose_delta[:2]  # x, y
    ego_pose_delta /= np.linalg.norm(ego_pose_delta)
    direction = np.array([ego_pose_delta[1], -ego_pose_delta[0], 0])  # y, -x
    return direction
