import torch
import numpy as np
import os
from street_gaussian.config import cfg
from street_gaussian.datasets.base_readers import fetchPly
from street_gaussian.models.gaussian_model import GaussianModel
from street_gaussian.utils.camera_utils import Camera
from street_gaussian.datasets.base_readers import get_Sphere_Norm

from easyvolcap.utils.console_utils import *

class GaussianModelBkgd(GaussianModel):
    def __init__(
        self,
        model_name='background',
        scene_center=np.array([0, 0, 0]),
        scene_radius=20,
    ):
        self.scene_center = torch.from_numpy(scene_center).float().cuda()
        self.scene_radius = torch.tensor([scene_radius]).float().cuda()
        num_classes = cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0
        self.background_mask = None

        super().__init__(model_name=model_name, num_classes=num_classes)

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        print('Create background model')
        pointcloud_path_bkgd = os.path.join(cfg.model_path, 'input_ply', 'points3D_bkgd.ply')
        assert os.path.exists(pointcloud_path_bkgd), f'Background pointcloud not found at {pointcloud_path_bkgd}'
        pcd = fetchPly(pointcloud_path_bkgd)

        sphere_normalization = get_Sphere_Norm(pcd.points)
        self.sphere_center = torch.from_numpy(sphere_normalization['center']).float().cuda()
        self.sphere_radius = torch.tensor([sphere_normalization['radius']]).float().cuda()

        return super().create_from_pcd(pcd, spatial_lr_scale)

    def set_background_mask(self, camera: Camera):
        pass
        # if cfg.mode == 'train':
        #     self.background_mask = None
        # else:
        #     # Base background mask (from Scaffold-GS, select Gaussian with radii > 0)
        #     self.background_mask = None
        #     rasterizer = make_rasterizer(camera)
        #     radii, means2D = rasterizer.visible_filter(
        #         means3D=self.get_xyz,
        #         scales=self.get_scaling,
        #         rotations=self.get_rotation,
        #     )
        #     radii_mask = radii > 0
        #     self.background_mask = radii_mask

        #     means2D = means2D[radii_mask]
        #     H, W = camera.image_height, camera.image_width
        #     pix_x = torch.clamp(means2D[:, 0].long(), min=0, max=W-1)
        #     pix_y = torch.clamp(means2D[:, 1].long(), min=0, max=H-1)

        #     self.background_mask[radii_mask] = camera.original_sky_mask[0, pix_y, pix_x]

        #     # debug
        #     import matplotlib.pyplot as plt
        #     image = camera.original_image.permute(1, 2, 0).cpu().numpy()
        #     x_value = pix_x.cpu().numpy()
        #     y_value = pix_y.cpu().numpy()
        #     plt.imshow(image)
        #     plt.scatter(x_value, y_value, s=1, c='r')
        #     plt.savefig('vis.png')

    @property
    def get_scaling(self):
        scaling = super().get_scaling
        return scaling if self.background_mask is None else scaling[self.background_mask]

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        return rotation if self.background_mask is None else rotation[self.background_mask]

    @property
    def get_xyz(self):
        xyz = super().get_xyz
        return xyz if self.background_mask is None else xyz[self.background_mask]

    @property
    def get_features(self):
        features = super().get_features
        return features if self.background_mask is None else features[self.background_mask]

    @property
    def get_opacity(self):
        opacity = super().get_opacity
        return opacity if self.background_mask is None else opacity[self.background_mask]

    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        max_grad = cfg.optim.get('densify_grad_threshold_bkgd', max_grad)
        if cfg.optim.get('densify_grad_abs_bkgd', False):
            grads = self.xyz_gradient_accum[:, 1:2] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        grads[grads.isnan()] = 0.0
        self.scalar_dict.clear()
        self.tensor_dict.clear()
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]
        # print('=' * 20)
        # print(f'Model name: {self.model_name}')
        # print(f'Number of 3d gaussians: {self.get_xyz.shape[0]}')

        # Clone and Split
        extent = self.scene_radius
        self.densify_and_clone(grads, max_grad, extent)
        bg_max_pts = cfg.optim.get('bkgd_max_points', 1e7)
        if len(self.get_xyz) < bg_max_pts:
            self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print(f'Prune points below min_opactiy: {prune_mask.sum()}')
        self.scalar_dict['points_below_min_opacity'] = prune_mask.sum().item()

        # Prune big points in world space
        if prune_big_points:
            dists = torch.linalg.norm(self.get_xyz - self.sphere_center, dim=1)
            # extent = torch.where(dists <= self.sphere_radius, self.scene_radius, dists / self.sphere_radius * self.scene_radius)
            big_points_ws = torch.max(self.get_scaling, dim=1).values > extent * self.percent_big_ws
            big_points_ws[dists > self.sphere_radius] = False
            # print('big points mean distance to sphere center', dists_big_mean, 'sphere_radius', self.sphere_radius)
            # print('before', big_points_ws.sum())
            # big_points_ws = torch.where(dists > 2 * self.sphere_radius, False, big_points_ws)
            # print('after', big_points_ws.sum())

            prune_mask = torch.logical_or(prune_mask, big_points_ws)

            # print(f'Prune big points in world space: {big_points_ws.sum()}')
            self.scalar_dict['points_big_ws'] = big_points_ws.sum().item()

        # Prune
        if self.max_screen_size:
            big_points_vs = self.max_radii2D > self.max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            # print(f'Prune big points in screen space: {big_points_vs.sum()}')

        # print(f'Prune mask: {prune_mask.sum()}')
        self.scalar_dict['points_pruned'] = prune_mask.sum().item()
        self.prune_points(prune_mask)

        # Reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()

        return self.scalar_dict, self.tensor_dict
