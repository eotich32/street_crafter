import torch
import torch.nn as nn
import numpy as np
import os
from bidict import bidict
from plyfile import PlyData, PlyElement

from street_gaussian.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion
from street_gaussian.utils.system_utils import mkdir_p
from street_gaussian.models.gaussian_model import GaussianModel
from street_gaussian.models.gaussian_model_bkgd import GaussianModelBkgd
from street_gaussian.models.gaussian_model_actor import GaussianModelActor
from street_gaussian.models.gaussian_model_sky import GaussianModelSky
from street_gaussian.utils.camera_utils import Camera
from street_gaussian.models.actor_pose import ActorPose
from street_gaussian.models.sky_cubemap import SkyCubeMap
from street_gaussian.models.color_correction import ColorCorrection
from street_gaussian.models.camera_pose import PoseCorrection
from street_gaussian.config import cfg

from easyvolcap.utils.console_utils import *


class StreetGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        self.include_background = cfg.model.nsg.get('include_bkgd', True)
        self.include_obj = cfg.model.nsg.get('include_obj', True)
        self.include_sky = cfg.model.nsg.get('include_sky', True)
        assert cfg.data.white_background is False if self.include_sky else True
        self.include_cube_map = cfg.model.nsg.get('include_cube_map', False)
        assert not (self.include_sky and self.include_cube_map)

        # fourier sh dimensions
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)

        # layer color correction
        self.use_color_correction = cfg.model.use_color_correction

        # camera pose optimizations (not test)
        self.use_pose_correction = cfg.model.use_pose_correction

        # symmetry
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1
        self.flip_matrix = torch.eye(3).float().to('cuda', non_blocking=True) * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        self.setup_functions()

    def set_visibility(self, include_list):
        self.include_list = include_list  # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                return True
            else:
                return False
        elif model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.create_from_pcd(pcd, spatial_lr_scale)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModel = getattr(self, model_name)
            plydata = model.make_ply()
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            plydata_list.append(plydata)

        PlyData(plydata_list).write(path)

    def save_ply_only_background(self, path):
        mkdir_p(os.path.dirname(path))
        model: GaussianModel = getattr(self, 'background')
        plydata = model.make_ply()
        el = PlyElement.describe(plydata, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:]  # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements

        self.active_sh_degree = self.max_sh_degree

    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])

        if self.actor_pose is not None:
            self.actor_pose.load_state_dict(state_dict['actor_pose'])

        if self.sky_cubemap is not None:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])

        if self.color_correction is not None:
            self.color_correction.load_state_dict(state_dict['color_correction'])

        if self.pose_correction is not None:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])

    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)

        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(is_final)

        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)

        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)

        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)

        return state_dict

    def setup_functions(self):
        camera_tracklets = self.metadata['camera_tracklets']
        obj_info = self.metadata['obj_meta']
        camera_timestamps = self.metadata['camera_timestamps']

        self.model_name_id = bidict()
        self.obj_list = []
        self.models_num = 0

        # Build background model
        if self.include_background:
            self.background = GaussianModelBkgd(
                model_name='background',
                scene_center=self.metadata['scene_center'],
                scene_radius=self.metadata['scene_radius'],
            )

            self.model_name_id['background'] = 0
            self.models_num += 1

        # Build object model
        if self.include_obj:
            for obj_id, obj_meta in obj_info.items():
                model_name = f'obj_{obj_id:03d}'
                setattr(self, model_name, GaussianModelActor(model_name=model_name, obj_meta=obj_meta))
                self.model_name_id[model_name] = self.models_num
                self.obj_list.append(model_name)
                self.models_num += 1

        # Build sky model            
        self.sky_cubemap = None
        if self.include_sky:
            self.sky = GaussianModelSky(model_name='sky')
            self.model_name_id['sky'] = self.models_num
            self.models_num += 1
        elif self.include_cube_map:
            self.sky_cubemap: Optional[SkyCubeMap] = SkyCubeMap()

        # Build actor model
        self.actor_pose: Optional[ActorPose] = ActorPose(camera_tracklets, camera_timestamps, obj_info) if self.include_obj else None

        # Build color correction
        self.color_correction: Optional[ColorCorrection] = ColorCorrection(self.metadata) if self.use_color_correction else None

        # Build pose correction
        self.pose_correction: Optional[PoseCorrection] = PoseCorrection(self.metadata) if self.use_pose_correction else None

    def parse_camera(self, camera: Camera):
        # set camera
        self.viewpoint_camera = camera

        # set background mask
        self.background.set_background_mask(camera)

        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0
        self.graph_gaussian_range = dict()
        idx = 0

        # background
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_bkgd
            self.graph_gaussian_range['background'] = [idx, idx + num_gaussians_bkgd]
            idx += num_gaussians_bkgd

        # object (build scene graph)
        self.graph_obj_list = []
        if self.include_obj:
            for i, obj_name in enumerate(self.obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                start_frame, end_frame = obj_model.start_frame, obj_model.end_frame
                if self.frame >= start_frame and self.frame <= end_frame and self.get_visibility(obj_name):
                    self.graph_obj_list.append(obj_name)
                    num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
                    self.num_gaussians += num_gaussians_obj
                    self.graph_gaussian_range[obj_name] = [idx, idx + num_gaussians_obj]
                    idx += num_gaussians_obj
                    
        # sky
        if self.get_visibility('sky'):
            num_gaussians_sky = self.sky.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_sky
            self.graph_gaussian_range['sky'] = [idx, idx + num_gaussians_sky]
            idx += num_gaussians_sky

        if len(self.graph_obj_list) > 0:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                obj_id = obj_model.obj_id
                with torch.set_grad_enabled(not camera.meta['is_novel_view']):
                    obj_rot = self.actor_pose.get_tracking_rotation(obj_id, self.viewpoint_camera)  # type: ignore
                    obj_trans = self.actor_pose.get_tracking_translation(obj_id, self.viewpoint_camera)  # type: ignore

                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)

                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)

            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)

            if cfg.mode == 'train':
                self.flip_mask = []
                for obj_name in self.graph_obj_list:
                    obj_model: GaussianModelActor = getattr(self, obj_name)
                    if obj_model.deformable or self.flip_prob == 0:
                        flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
                    else:
                        flip_mask = torch.rand_like(obj_model.get_xyz[:, 0]) < self.flip_prob
                    self.flip_mask.append(flip_mask)
                self.flip_mask = torch.cat(self.flip_mask, dim=0)

    @property
    def get_scaling(self):
        scalings = []

        if self.get_visibility('background'):
            scaling_bkgd = self.background.get_scaling
            scalings.append(scaling_bkgd)

        for obj_name in self.graph_obj_list:
            if self.get_visibility(obj_name):
                obj_model: GaussianModelActor = getattr(self, obj_name)

                scaling = obj_model.get_scaling

                scalings.append(scaling)
            
        if self.get_visibility('sky'):
            scaling_sky = self.sky.get_scaling
            scalings.append(scaling_sky)

        scalings = torch.cat(scalings, dim=0)
        return scalings

    @property
    def get_rotation(self):
        rotations = []

        if self.get_visibility('background'):
            rotations_bkgd = self.background.get_rotation
            rotations.append(rotations_bkgd)

        if len(self.graph_obj_list) > 0:
            rotations_local = []
            for i, obj_name in enumerate(self.graph_obj_list):
                if self.get_visibility(obj_name):
                    obj_model: GaussianModelActor = getattr(self, obj_name)
                    rotation_local = obj_model.get_rotation
                    rotations_local.append(rotation_local)

            if len(rotations_local) > 0:    
                rotations_local = torch.cat(rotations_local, dim=0)
                if cfg.mode == 'train':
                    rotations_local = rotations_local.clone()
                    rotations_flip = rotations_local[self.flip_mask]
                    if len(rotations_flip) > 0:
                        rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_flip)
                rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)  # type: ignore
                rotations_obj = torch.nn.functional.normalize(rotations_obj)
                rotations.append(rotations_obj)
            
        if self.get_visibility('sky'):
            rotations_sky = self.sky.get_rotation
            rotations.append(rotations_sky)

        rotations = torch.cat(rotations, dim=0)
        return rotations

    @property
    def get_xyz(self):
        xyzs = []
        if self.get_visibility('background'):
            xyz_bkgd = self.background.get_xyz
            xyzs.append(xyz_bkgd)

        if len(self.graph_obj_list) > 0:
            xyzs_local = []

            for i, obj_name in enumerate(self.graph_obj_list):
                if self.get_visibility(obj_name):
                    obj_model: GaussianModelActor = getattr(self, obj_name)
                    xyz_local = obj_model.get_xyz
                    xyzs_local.append(xyz_local)

            if len(xyzs_local) > 0:
                xyzs_local = torch.cat(xyzs_local, dim=0)
                if cfg.mode == 'train':
                    xyzs_local = xyzs_local.clone()
                    xyzs_local[self.flip_mask, self.flip_axis] *= -1
                obj_rots = quaternion_to_matrix(self.obj_rots)
                xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
                xyzs.append(xyzs_obj)

        if self.get_visibility('sky'):
            xyz_sky = self.sky.get_xyz
            xyzs.append(xyz_sky)

        if not len(xyzs):
            xyzs = torch.zeros(0, 3, device="cuda")
        else:
            xyzs = torch.cat(xyzs, dim=0)

        return xyzs

    @property
    def get_features(self):
        features = []

        if self.get_visibility('background'):
            features_bkgd = self.background.get_features
            features.append(features_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            if self.get_visibility(obj_name):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                feature_obj = obj_model.get_features_fourier(self.frame)
                features.append(feature_obj)
            
        if self.get_visibility('sky'):
            features_sky = self.sky.get_features
            features.append(features_sky)

        features = torch.cat(features, dim=0)

        return features

    @property
    def get_opacity(self):
        opacities = []

        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        for obj_name in self.graph_obj_list:
            if self.get_visibility(obj_name):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                opacity = obj_model.get_opacity
                opacities.append(opacity)
            
        if self.get_visibility('sky'):
            opacity_sky = self.sky.get_opacity
            opacities.append(opacity_sky)

        opacities = torch.cat(opacities, dim=0)
        return opacities

    def get_covariance(self, scaling_modifier=1):
        scaling = self.get_scaling  # [N, 1]
        rotation = self.get_rotation  # [N, 4]
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: GaussianModel = getattr(self, model_name)
            model.oneupSHdegree()

        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.training_setup()

        if self.actor_pose is not None:
            self.actor_pose.training_setup()

        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()

        if self.color_correction is not None:
            self.color_correction.training_setup()

        if self.pose_correction is not None:
            self.pose_correction.training_setup()

    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)

        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)

        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)

        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)

        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)

    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_optimizer()

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()

        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()

        if self.color_correction is not None:
            self.color_correction.update_optimizer()

        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    @torch.no_grad()
    def set_max_radii2D(self, radii, visibility_filter):
        radii = radii.float()

        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            max_radii2D_model = radii[start:end]
            model.max_radii2D[visibility_model] = torch.max(
                model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
            
    @torch.no_grad()
    def set_max_radii2D_sky(self, radii, visibility_filter):
        radii = radii.float()
        self.sky.max_radii2D[visibility_filter] = torch.max(self.sky.max_radii2D[visibility_filter], radii[visibility_filter])

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, visibility_filter, viewpoint_cam=None):
        if hasattr(viewspace_point_tensor, 'absgrad'):
            viewspace_point_tensor_grad = torch.cat([viewspace_point_tensor.absgrad, viewspace_point_tensor.grad], dim=-1)
            if viewspace_point_tensor_grad.ndim == 3: viewspace_point_tensor_grad = viewspace_point_tensor_grad[0]
            viewspace_point_tensor_grad = viewspace_point_tensor_grad * 0.5 * torch.as_tensor([viewpoint_cam.image_width, viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.image_height]).to(viewspace_point_tensor_grad, non_blocking=True)
        else:
            viewspace_point_tensor_grad = viewspace_point_tensor.grad

        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
            model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)
            model.denom[visibility_model] += 1

    @torch.no_grad()
    def add_densification_stats_sky(self, viewspace_point_tensor, visibility_filter, viewpoint_cam=None):
        if hasattr(viewspace_point_tensor, 'absgrad'):
            viewspace_point_tensor_grad = torch.cat([viewspace_point_tensor.absgrad, viewspace_point_tensor.grad], dim=-1)
            if viewspace_point_tensor_grad.ndim == 3: viewspace_point_tensor_grad = viewspace_point_tensor_grad[0]
            viewspace_point_tensor_grad = viewspace_point_tensor_grad * 0.5 * torch.as_tensor([viewpoint_cam.image_width, viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam.image_height]).to(viewspace_point_tensor_grad, non_blocking=True)
        else:
            viewspace_point_tensor_grad = viewspace_point_tensor.grad

        visibility_model = visibility_filter
        self.sky.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad[visibility_model, :2], dim=-1, keepdim=True)
        self.sky.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad[visibility_model, 2:], dim=-1, keepdim=True)
        self.sky.denom[visibility_model] += 1

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)

            scalars_, tensors_ = model.densify_and_prune(max_grad, min_opacity, prune_big_points) # type: ignore
            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_

        return scalars, tensors

    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss

    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()
