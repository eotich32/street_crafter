import numpy as np
import torch
import copy
import math
from PIL import Image
from tqdm import tqdm
from street_gaussian.utils.general_utils import PILtoTorch, NumpytoTorch, matrix_to_quaternion
from street_gaussian.utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2, getProjectionMatrixK
from street_gaussian.datasets.base_readers import CameraInfo
from street_gaussian.config import cfg

# if training, put everything to cuda
# image_to_cuda = (cfg.mode == 'train')

from easyvolcap.utils.console_utils import *


class Camera():
    def __init__(
        self,
        id,
        R, T,
        FoVx, FoVy, K,
        image, image_name,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        metadata=dict(),
        guidance=dict(),
    ):
        self.id = id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.trans, self.scale = trans, scale

        # metadata
        self.meta = metadata

        # guidance
        self.guidance = guidance
        self.original_image = image.clamp(0., 1.)

        self.image_height, self.image_width = self.original_image.shape[1], self.original_image.shape[2]
        self.zfar = 1000.0
        self.znear = 0.001
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to('cuda', non_blocking=True)

        if self.K is not None:
            self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0, 1).to('cuda', non_blocking=True)
            self.K = torch.from_numpy(self.K).float().to('cuda', non_blocking=True)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to('cuda', non_blocking=True)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # if 'ego_pose' in self.meta.keys():
        #     self.ego_pose = torch.from_numpy(self.meta['ego_pose']).float().to('cuda', non_blocking=True)
        #     del self.meta['ego_pose']

        # if 'extrinsic' in self.meta.keys():
        #     self.extrinsic = torch.from_numpy(self.meta['extrinsic']).float().to('cuda', non_blocking=True)
        #     del self.meta['extrinsic']

    def set_extrinsic(self, ext: torch.Tensor, world2cam=False):
        w2c = ext if world2cam else torch.linalg.inv(ext)

        # set R, T
        self.R = w2c[:3, :3].T.detach().cpu().numpy()
        self.T = w2c[:3, 3].detach().cpu().numpy()

        # change attributes associated with R, T
        self.world_view_transform[:3, :3] = w2c[:3, :3].T
        self.world_view_transform[3, :3] = w2c[:3, 3]
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def set_intrinsic(self, K: torch.Tensor):
        self.K = K
        self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0, 1).to('cuda', non_blocking=True)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def get_extrinsic(self, world2cam=False) -> torch.Tensor:
        w2c = self.world_view_transform.transpose(0, 1)
        return w2c if world2cam else torch.linalg.inv(w2c)

    def get_intrinsic(self) -> torch.Tensor:
        return self.K

    def set_device(self, device):
        self.original_image = self.original_image.to(device)
        for k, v in self.guidance.items():
            self.guidance[k] = v.to(device, non_blocking=True)

        # Here we crop top of the image for driving scenes

    # def get_extrinsic(self):
    #     w2c = np.eye(4)
    #     w2c[:3, :3] = self.R.T
    #     w2c[:3, 3] = self.T
    #     c2w = np.linalg.inv(w2c)
    #     return c2w

    # def get_intrinsic(self):
    #     ixt = self.K.cpu().numpy()
    #     return ixt


# 专门用于加载场景时渲染diffusion的雷达条件图
# 原本的代码结构有循环依赖，对于新视角来说，要用WaymoPointCloudProcessor渲染雷达图，必须先生成Camera对象。而生成Camera对象，又需要加载image，而这个image就是渲染出来的雷达图。
# 原本的代码在waymo_processor包中有专门的渲染雷达图的代码，需要在加载场景之前事先执行
# 这里改成直接在加载场景时自动渲染，一步完成。
class Camera4ForRenderCondition():
    def __init__(
        self,
        R, T,
        FoVx, FoVy, K, image_height, image_width,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        metadata=dict(),
    ):
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.trans, self.scale = trans, scale
        self.meta = metadata
        self.image_height, self.image_width = image_height, image_width
        self.zfar = 1000.0
        self.znear = 0.001
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to('cuda', non_blocking=True)

        if self.K is not None:
            self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0, 1).to('cuda', non_blocking=True)
            self.K = torch.from_numpy(self.K).float().to('cuda', non_blocking=True)
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to('cuda', non_blocking=True)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_extrinsic(self, world2cam=False) -> torch.Tensor:
        w2c = self.world_view_transform.transpose(0, 1)
        return w2c if world2cam else torch.linalg.inv(w2c)

    def get_intrinsic(self) -> torch.Tensor:
        return self.K

    def set_device(self, device):
        self.original_image = self.original_image.to(device)
        for k, v in self.guidance.items():
            self.guidance[k] = v.to(device, non_blocking=True)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def loadguidance(guidance, resolution):
    new_guidance = dict()
    for k, v in guidance.items():
        if k == 'mask':
            new_guidance['mask'] = PILtoTorch(v, resolution, resize_mode=Image.NEAREST).bool()
        elif k == 'acc_mask':
            new_guidance['acc_mask'] = PILtoTorch(v, resolution, resize_mode=Image.NEAREST).bool()
        elif k == 'sky_mask':
            new_guidance['sky_mask'] = PILtoTorch(v, resolution, resize_mode=Image.NEAREST).bool()
        elif k == 'obj_bound':
            new_guidance['obj_bound'] = PILtoTorch(v, resolution, resize_mode=Image.NEAREST).bool()
        elif k == 'lidar_depth':
            new_guidance['lidar_depth'] = NumpytoTorch(v, resolution, resize_mode=Image.NEAREST).float()

    return new_guidance


WARNED = False


def loadCam(cam_info: CameraInfo, resolution_scale, scale=1.0):
    orig_w = cam_info.width
    orig_h = cam_info.height
    # if cfg.mode != 'train':
    scale = min(scale, 1600 / orig_w)
    scale = scale / resolution_scale
    resolution = (int(orig_w * scale), int(orig_h * scale))

    K = copy.deepcopy(cam_info.K)
    K[:2] *= scale

    image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
    guidance = loadguidance(cam_info.guidance, resolution)

    return Camera(
        id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        K=K,
        image=image,
        image_name=cam_info.image_name,
        metadata=cam_info.metadata,
        guidance=guidance,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, scale=1.0):
    camera_list = []

    for i, cam_info in enumerate(tqdm(cam_infos)):
        camera_list.append(loadCam(cam_info, resolution_scale, scale))

    return camera_list


def camera_to_JSON(id, camera: CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def make_rasterizer(
    viewpoint_camera: Camera,
    active_sh_degree=0,
    bg_color=None,
    scaling_modifier=None,
):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    if bg_color is None:
        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().to('cuda', non_blocking=True)
    if scaling_modifier is None:
        scaling_modifier = cfg.render.scaling_modifier
    debug = cfg.render.debug

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
    )

    rasterizer: GaussianRasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer
