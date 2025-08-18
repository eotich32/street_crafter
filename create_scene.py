from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.datasets.dataset import Dataset, sceneLoadTypeCallbacks
from street_gaussian.pointcloud_processor import getPointCloudProcessor
from street_gaussian.models.scene import Scene
from street_gaussian.config import cfg
from video_diffusion.sample_condition import VideoDiffusionModel
from street_gaussian.utils.camera_utils import cameraList_from_camInfos

def create_scene(device='cuda'):
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.metadata)
    if (cfg.mode == 'train' and cfg.diffusion.use_diffusion) or cfg.mode in ['diffusion','parallel_diffusion']:
        if cfg.diffusion.use_img_diffusion:
            print("Image Diffusion Model is used")
            video_diffusion = None
        else:
            print("Lodaing Video Diffusion Model...")
            video_diffusion = VideoDiffusionModel(
                config_path=cfg.diffusion.config_path,
                ckpt_path=cfg.diffusion.ckpt_path,
                height=cfg.diffusion.height,
                width=cfg.diffusion.width
            )
    else:
        video_diffusion = None

    if cfg.mode == 'train' or cfg.mode in ['diffusion','parallel_diffusion']:
        pointcloud_processor = getPointCloudProcessor()
    else:
        pointcloud_processor = None

    scene = Scene(
        gaussians=gaussians,
        dataset=dataset,
        pointcloud_processor=pointcloud_processor,
        diffusion=video_diffusion,
        device=device
    )
    return scene


def create_scene_for_diffusion_inference(scene_meta, device='cuda'):
    gaussians = StreetGaussianModel(scene_meta)
    if cfg.diffusion.use_img_diffusion:
        print("Image Diffusion Model is used")
        video_diffusion = None
    else:
        video_diffusion = VideoDiffusionModel(
            config_path=cfg.diffusion.config_path,
            ckpt_path=cfg.diffusion.ckpt_path,
            height=cfg.diffusion.height,
            width=cfg.diffusion.width
        )
        print("Diffusion model loaded")
    pointcloud_processor = getPointCloudProcessor()

    scene = Scene(
        gaussians=gaussians,
        dataset=None,
        pointcloud_processor=pointcloud_processor,
        diffusion=video_diffusion,
        device=device
    )

    dataset_type = cfg.data.get('type')
    assert dataset_type in sceneLoadTypeCallbacks.keys(), 'Could not recognize scene type!'
    scene_info = sceneLoadTypeCallbacks[dataset_type](cfg.source_path, **cfg.data)
    # print(f'=================== len(scene_info.train_cameras): {len(scene_info.train_cameras)}')
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1)
    return scene, train_cameras
