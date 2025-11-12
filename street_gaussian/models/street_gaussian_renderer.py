import torch
from street_gaussian.utils.sh_utils import eval_sh
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.utils.camera_utils import Camera, make_rasterizer
from street_gaussian.config import cfg

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.utils.timer_utils import timer

class StreetGaussianRenderer():
    def __init__(
        self,
    ):
        self.cfg = cfg.render

    def render_all(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):

        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, parse_camera_again=parse_camera_again)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, parse_camera_again=parse_camera_again)

        # render object
        render_object = self.render_object(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, parse_camera_again=parse_camera_again)

        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']

        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)

        return result

    def render_object(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=pc.obj_list)
        if parse_camera_again: pc.parse_camera(viewpoint_camera)

        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result

    def render_background(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=['background'])
        if parse_camera_again: pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result

    def render_sky(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        parse_camera_again: bool = True,
    ):
        pc.set_visibility(include_list=['sky'])
        if parse_camera_again: pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        return result

    def render(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        exclude_list=[],
        parse_camera_again: bool = True,
        collect_timing: bool = False,
    ):
        exclude_list = exclude_list + ['sky']
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))

        # Step1: render foreground
        pc.set_visibility(include_list)
        if parse_camera_again: pc.parse_camera(viewpoint_camera)

        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # Step2: render sky
        if pc.include_sky:
            result_sky = self.render_sky(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, parse_camera_again=False)
            result['rgb'] = result['rgb'] + result_sky['rgb'] * (1 - result['acc'])
            result['viewspace_points_sky'] = result_sky['viewspace_points']
            result['visibility_filter_sky'] = result_sky['visibility_filter']
            result['radii_sky'] = result_sky['radii']
        elif pc.include_sky_cubemap:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach()) # type: ignore
            # sky_color = pc.color_correction(viewpoint_camera, sky_color, use_sky=True) if use_color_correction else sky_color
            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb']) # type: ignore

        if cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)
        
        return result

    def render_novel_view(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        exclude_list=[],
        parse_camera_again: bool = True,
    ):  
        exclude_list = exclude_list + ['sky']
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
        pc.set_visibility(include_list)
        if parse_camera_again: pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        if pc.include_sky:
            result_sky = self.render_sky(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, parse_camera_again=False)
            result['rgb'] = result['rgb'] + result_sky['rgb'] * (1 - result['acc'])
        elif pc.include_sky_cubemap:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach()) # type: ignore
            # sky_color = pc.color_correction(viewpoint_camera, sky_color, use_sky=True) if use_color_correction else sky_color
            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result

    def render_kernel(self, *args, **kwargs):
        if self.cfg.use_gsplat:
            return self.render_kernel_gsplat(*args, **kwargs)
        else:
            return self.render_kernel_diff_gauss(*args, **kwargs)
        # return self.render_kernel_gsplat(*args, **kwargs)

    def render_kernel_gsplat(
        self,
        camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        white_background=cfg.data.white_background,
        tile_size: int = 16,
        use_depth: bool = True,
        absgrad: bool = True,
        antialiasing: bool = cfg.render.antialiasing,
    ):
        xyz3 = pc.get_xyz
        
        if len(xyz3) == 0:
            if white_background:
                rendered_color = torch.ones(3, int(camera.image_height), int(camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(camera.image_height), int(camera.image_width), device="cuda")

            rendered_acc = torch.zeros(1, int(camera.image_height), int(camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(camera.image_height), int(camera.image_width), device="cuda")

            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }
        
        import math
        from gsplat.rendering import rasterization, fully_fused_projection, isect_tiles, isect_offset_encode, rasterize_to_pixels, spherical_harmonics

        rgb3 = pc.get_features
        scale = pc.get_scaling
        quats = pc.get_rotation
        occ1 = pc.get_opacity

        width = camera.image_width
        height = camera.image_height

        # c2w = affine_padding(torch.cat([torch.as_tensor(camera.R[None]), torch.as_tensor(camera.T[None, ..., None])], dim=-1)).to(xyz3, non_blocking=True)
        w2c = torch.as_tensor(camera.world_view_transform.mT.to(xyz3, non_blocking=True))[None]  # 1, 4, 4
        K = torch.as_tensor(camera.K).to(xyz3, non_blocking=True)[None]

        # Project Gaussians to 2D
        proj_results = fully_fused_projection(
            xyz3,
            None,
            quats,
            scale,
            w2c,
            K,
            width,
            height,
            packed=False,
            near_plane=camera.znear,
            far_plane=camera.zfar,
            calc_compensations=antialiasing,
        )

        radii, means2d, depths, conics, compensations = proj_results
        opacities = occ1[None, :, 0]

        if compensations is not None:
            opacities = opacities * compensations

        # Identify intersecting tiles
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_cameras=1,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

        # Handle spherical harmonics
        dirs = xyz3[None, :, :] - camera.camera_center.to(xyz3, non_blocking=True)  # [1, N, 3]
        masks = radii > 0  # [1, N]
        shs = rgb3.expand(1, -1, -1, -1)  # [1, N, K, 3]
        colors = spherical_harmonics(pc.max_sh_degree, dirs, shs, masks=masks)  # [1, N, 3]
        colors = torch.clamp_min(colors + 0.5, 0.0)

        if cfg.mode == 'train' and means2d.requires_grad:
            means2d.retain_grad()

        if use_depth:
            colors = torch.cat((colors, depths[..., None]), dim=-1)
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=None,
            packed=False,
            absgrad=absgrad,
        )

        if use_depth:
            rendered_color = render_colors[..., :-1]
            rendered_depth = render_colors[..., -1:] / render_alphas.clamp(min=1e-10)
        else:
            rendered_color = render_colors
            rendered_depth = render_alphas
        rendered_acc = render_alphas

        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)

        result = {
            "rgb": rendered_color[0].permute(2, 0, 1),
            "acc": rendered_acc[..., 0],
            "depth": rendered_depth[..., 0],
            "viewspace_points": means2d,
            "visibility_filter": radii[0] > 0,
            "radii": radii[0] / float(max(height, width))
        }

        return result

    def render_kernel_diff_gauss(
        self,
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python=None,
        compute_cov3D_python=None,
        scaling_modifier=None,
        override_color=None,
        white_background=cfg.data.white_background,
    ):

        if pc.num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")

            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")

            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().to('cuda', non_blocking=True)
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)

        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if cfg.mode == 'train':
            screenspace_points = torch.zeros((pc.num_gaussians, 3), requires_grad=True, device='cuda', dtype=torch.float) + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            semantics=None,
        )

        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii
        }

        return result
