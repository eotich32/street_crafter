import numpy as np
import torch
import math
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import affine_inverse, affine_padding
from simple_knn._C import distCUDA2

def get_device(gpu_id=0):
    # Check if the GPU is available
    if torch.cuda.is_available():
        # Specify the device using the GPU ID
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print("CUDA is not available. Returning the tensor on CPU.")
        device = torch.device('cpu')

    return device



def render_pointcloud_pytorch3d(c2w, ixt, points, features, H, W, gpu_id=0):
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        PerspectiveCameras,
    )

    from pytorch3d.structures import Pointclouds
    
    device = get_device(gpu_id)

    if len(c2w) == 1:
        c2w = torch.as_tensor(c2w[0][None]).to(device, non_blocking=True).float()
        ixt = torch.as_tensor(ixt[0][None]).to(device, non_blocking=True).float()
        points_list = torch.as_tensor(points[0][None]).to(device, non_blocking=True).float()
        features_list = torch.as_tensor(features[0][None]).to(device, non_blocking=True).float()
    else:
        c2w = torch.as_tensor(c2w[None]).to(device, non_blocking=True).float()
        ixt = torch.as_tensor(ixt[None]).to(device, non_blocking=True).float()
        points_list = torch.as_tensor(points[None]).to(device, non_blocking=True).float()
        features_list = torch.as_tensor(features[None]).to(device, non_blocking=True).float()    


    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3:]

    fs = ixt[:, 0, 0]  # focal length
    c = ixt[:, :2, 2]  # cx, cy

    # change from opencv to pytorch3d camera
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], 2)  # from RDF to LUF for Rotation
    new_c2w = torch.cat([R, T], 2)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[[0, 0, 0, 1]]]).repeat(new_c2w.shape[0], 1, 1).to(device, non_blocking=True)), 1))
    R_new, T_new = w2c[:, :3, :3].permute(0, 2, 1), w2c[:, :3, 3]  # convert R to row-major matrix
    image_size = ((H, W),)  # (h, w)
    cameras = PerspectiveCameras(focal_length=fs, principal_point=c, in_ndc=False, image_size=image_size, R=R_new, T=T_new, device=device)
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=0.01,
        points_per_pixel=10,
        bin_size=0
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    pointcloud = Pointclouds(points=points_list, features=features_list).extend(1)

    # Render the point cloud
    result = renderer(pointcloud)

    # import cv2
    # x = result[0, ..., [2, 1, 0]].cpu().numpy()
    # x = (x * 255).astype(np.uint8)
    # cv2.imwrite('test.png', x)

    return result

def render_pointcloud_diff_point_rasterization(c2w, ixt, points, features, H, W, gpu_id=0, occ=1.0,
                                               scale=0.035,  # if not using ndc scale, this is the scale in world space, should use 0.025
                                               use_ndc_scale=False,
                                               use_knn_scale=False,
                                               knn_scale_down=1.0,
                                               ):
    """
    Render pointcloud using gsplat
    Requires points and features to have a batch dim to be discarded by us here
    """
    if len(c2w) == 1: c2w[0][None]
    else: c2w = c2w[None]
    if len(ixt) == 1: ixt[0][None]
    else: ixt = ixt[None]
    if len(points) == 1: points[0][None]
    else: points = points[None]
    if len(features) == 1: features[0][None]
    else: features = features[None]

    device = get_device(gpu_id)

    c2w = torch.as_tensor(c2w).to(device, non_blocking=True).float()[0]
    w2c = affine_inverse(c2w)
    cpu_ixt = ixt[0]
    ixt = torch.as_tensor(ixt).to(device, non_blocking=True).float()[0]
    xyz3 = torch.as_tensor(points).to(device, non_blocking=True).float()[0]
    rgb3 = torch.as_tensor(features).to(device, non_blocking=True).float()[0, ..., :3]  # only rgb

    occ1 = torch.full_like(xyz3[..., :1], occ)
    scales = torch.full_like(xyz3[..., :1], scale)  # isometrical scale
    quats = xyz3.new_zeros(xyz3.shape[:-1] + (4,))  # identity quaternion
    quats[..., 3] = 1

    if use_ndc_scale:
        # Revert the ndc scale back to world space
        # gl_PointSize = abs(H * K[1][1] * radius / gl_Position.w) * radii_mult;  // need to determine size in pixels
        views = xyz3 @ w2c[:3, :3].mT + w2c[:3, 3]
        x, y, z = views.chunk(3, dim=-1)
        scales = scales * z / ixt[0, 0] 
        scales = scales * 0.5 * H if H <= W else scales * 0.5 * W
    elif use_knn_scale:
        # Use KNN to determine point scale based on local point density
        dist2 = torch.clamp_min(distCUDA2(xyz3), 0.0000001) 
        scales = torch.sqrt(dist2)[..., None] * knn_scale_down
        scales = torch.minimum(scales, torch.full_like(scales, scale))

    from diff_point_rasterization import PointRasterizationSettings, PointRasterizer

    from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
    gaussian_camera = convert_to_gaussian_camera(
        ixt,
        w2c[:3, :3],
        w2c[:3, 3:],
        torch.as_tensor(H).to(xyz3.device, non_blocking=True),
        torch.as_tensor(W).to(xyz3.device, non_blocking=True),
        torch.as_tensor(1.0).to(xyz3.device, non_blocking=True),
        torch.as_tensor(100.0).to(xyz3.device, non_blocking=True),
        cpu_ixt,
        None,
        None,
        torch.as_tensor(H),
        torch.as_tensor(W),
        torch.as_tensor(1.0),
        torch.as_tensor(100.0)
    )

    # Prepare rasterization settings for gaussian
    raster_settings = PointRasterizationSettings(
        image_height=gaussian_camera.image_height,
        image_width=gaussian_camera.image_width,
        tanfovx=gaussian_camera.tanfovx,
        tanfovy=gaussian_camera.tanfovy,
        bg=torch.full([3], 0.0, device=xyz3.device),  # GPU
        scale_modifier=1.0,
        viewmatrix=gaussian_camera.world_view_transform,
        projmatrix=gaussian_camera.full_proj_transform,
        sh_degree=0,
        max_hit=10,
        campos=gaussian_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = torch.zeros_like(xyz3, requires_grad=True) + 0  # gradient magic
    if scr.requires_grad: scr.retain_grad()

    rasterizer = PointRasterizer(raster_settings=raster_settings)
    rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
        means3D=xyz3,
        means2D=scr,
        colors_precomp=rgb3,
        opacities=occ1,
        radius=scales,
    )

    rgb = rendered_image[None].permute(0, 2, 3, 1)
    acc = rendered_alpha[None].permute(0, 2, 3, 1)
    dpt = rendered_depth[None].permute(0, 2, 3, 1)

    return torch.cat([rgb, acc], dim=-1)  # 1, H, W, 4

