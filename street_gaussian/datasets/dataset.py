from easyvolcap.utils.console_utils import *
import random
from street_gaussian.utils.camera_utils import cameraList_from_camInfos
from street_gaussian.config import cfg
from street_gaussian.datasets.base_readers import SceneInfo
from street_gaussian.datasets.waymo_readers import readWaymoInfo
from street_gaussian.datasets.colmap_readers import readColmapSceneInfo

sceneLoadTypeCallbacks = {
    "Waymo": readWaymoInfo,
    "Colmap": readColmapSceneInfo,
}


class Dataset():
    def __init__(self):
        self.cfg = cfg.data
        self.model_path = cfg.model_path
        self.source_path = cfg.source_path
        self.images = self.cfg.images

        self.train_cameras = {}
        self.test_cameras = {}
        self.novel_view_cameras = {}

        dataset_type = cfg.data.get('type')
        assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
        scene_info: SceneInfo = sceneLoadTypeCallbacks[dataset_type](self.source_path, **cfg.data)

        self.metadata = scene_info.metadata

        if self.cfg.shuffle and cfg.mode == 'train':
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        for resolution_scale in cfg.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale)

            if scene_info.novel_view_cameras is not None:
                print("Loading Novel View Cameras")
                self.novel_view_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.novel_view_cameras, resolution_scale)

    def getmeta(self, k):
        return self.metadata[k]
