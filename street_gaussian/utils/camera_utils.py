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


class PseudoCamera():
    def __init__(self, R, T, K, FoVx, FoVy, width, height, meta):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        self.meta = meta

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


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

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def normalize(x):
    return x / np.linalg.norm(x)


def generate_random_poses_360(views, n_frames=10000, z_variation=0.1, z_phase=0):

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)


    def view_matrix(lookdir, up, position, subtract_position=False):
      """Construct lookat view matrix."""
      vec2 = normalize((lookdir - position) if subtract_position else lookdir)
      vec0 = normalize(np.cross(up, vec2))
      vec1 = normalize(np.cross(vec2, vec0))
      m = np.stack([vec0, vec1, vec2, position], axis=1)
      return m

    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    theta = np.random.rand(n_frames) * 2. * np.pi
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = view_matrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses