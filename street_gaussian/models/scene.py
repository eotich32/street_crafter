import os
import torch
from typing import Union, Optional
from street_gaussian.datasets.dataset import Dataset
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.config import cfg
from street_gaussian.utils.system_utils import searchForMaxIteration
from street_gaussian.pointcloud_processor.base_processor import BasePointCloudProcessor
from video_diffusion.sample_condition import VideoDiffusionModel

from easyvolcap.utils.console_utils import *


class Scene:
    def __init__(self,
                 dataset: Dataset,
                 gaussians: StreetGaussianModel,  # StreetGaussian Model
                 diffusion: Optional[VideoDiffusionModel] = None,  # VideoDiffusion Model
                 pointcloud_processor: Optional[BasePointCloudProcessor] = None,  # Pointcloud Processor
                 device='cuda',
        ):
        self.dataset = dataset
        self.gaussians = gaussians

        # We only need diffusion model and pointcloud processor during training
        self.diffusion = diffusion
        self.pointcloud_processor = pointcloud_processor

        if cfg.mode == 'diffusion':
            assert self.pointcloud_processor is not None
        elif cfg.mode == 'train':
            assert self.pointcloud_processor is not None
            self.pointcloud_processor.initailize_ply(cfg.model_path, self.dataset.getmeta('obj_meta'))

            print("Creating gaussian model from point cloud")
            self.gaussians.create_from_pcd(pcd=None, spatial_lr_scale=self.dataset.getmeta('scene_radius'))

            # # Move training images to GPU
            # for camera in list(self.getTrainCameras() + self.getNovelViewCameras()):
            #     camera.set_device('cuda')
        else:
            # First check if there is a checkpoint saved and get the iteration to load from
            assert (os.path.exists(cfg.trained_model_dir))
            if cfg.loaded_iter == -1:
                self.loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
            else:
                self.loaded_iter = cfg.loaded_iter

            # Load checkpoint if it exists
            # if cfg.mode != 'condition' and cfg.mode != 'diffusion':
            print("Loading checkpoint at iteration {}".format(self.loaded_iter))
            checkpoint_path = os.path.join(cfg.trained_model_dir, f"iteration_{str(self.loaded_iter)}.pth")
            assert os.path.exists(checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.gaussians.load_state_dict(state_dict=state_dict)

        # render conditions
        if (cfg.mode == 'train' and cfg.diffusion.use_diffusion) or cfg.mode == 'diffusion':
            #assert self.diffusion is not None
            cameras = self.getTrainCameras() + self.getTestCameras() + self.getNovelViewCameras()
            print('Rendering LiDAR conditions')
            self.pointcloud_processor.render_conditions(cameras, self.dataset.getmeta('obj_meta'))  # type: ignore

    def getTrainCameras(self, scale=1):
        return self.dataset.train_cameras[scale]

    def getTestCameras(self, scale=1):
        return self.dataset.test_cameras[scale]

    def getNovelViewCameras(self, scale=1):
        return self.dataset.novel_view_cameras[scale]
