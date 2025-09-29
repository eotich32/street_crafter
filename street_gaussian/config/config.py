from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from street_gaussian.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.environ['PWD']
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train'
cfg.task = 'hello'  # task folder name
cfg.exp_name = 'test'  # experiment folder name
cfg.gpus = [-1]  # list of gpus to use
cfg.debug = False
cfg.resume = True  # If set to True, resume training from the last checkpoint.
cfg.no_date = True
cfg.diffusion_parallel = 0

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

cfg.eval = CN()
cfg.eval.skip_train = False
cfg.eval.skip_test = False
cfg.eval.skip_novel = False
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.eval_novel = False
cfg.eval.quiet = False
cfg.eval.visualize = False
cfg.eval.time_skip_frames = 30

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [7000, 30000]
cfg.train.save_iterations = [7000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False
cfg.train.novel_view_prob = 0.4
cfg.train.reg_obj_acc_every = 5

cfg.optim = CN()
# learning rate
cfg.optim.position_lr_init = 0.00016  # position_lr_init_{bkgd, obj ...}, similar to the following
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002  # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.densify_grad_abs_bkgd = False  # densification strategy from AbsGS
cfg.optim.densify_grad_abs_obj = False  # densification strategy from AbsGS
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
cfg.optim.prune_big_points = False
cfg.optim.max_screen_size = 1.0  # 0.5 of the full resolution

# loss weight
cfg.optim.lambda_l1 = 1.
cfg.optim.lambda_lpips = 0.01
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.
cfg.optim.lambda_reg = 0.
cfg.optim.lambda_depth_lidar = 0.
cfg.optim.lambda_depth_mono = 0.
cfg.optim.lambda_normal_mono = 0.
cfg.optim.lambda_color_correction = 0.
cfg.optim.lambda_pose_correction = 0.
cfg.optim.lambda_scale_flatten = 0.
cfg.optim.lambda_opacity_sparse = 0.

cfg.optim.lambda_novel = 0.1
cfg.optim.lambda_novel_l1 = 0.1
cfg.optim.lambda_novel_lpips = 1.0
cfg.optim.lambda_novel_dssim = 0.1

cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1  # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.
cfg.model.gaussian.flip_prob = 0.  # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True  # include background
cfg.model.nsg.include_obj = True  # include object
cfg.model.nsg.include_sky = True  # include sky
cfg.model.nsg.include_cube_map = False  # include cube map
cfg.model.nsg.opt_track = False  # tracklets optimization
cfg.model.sky = CN()
cfg.model.sky.use_cube_map = False
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True


# Note: these algorithms are not fully tested.
cfg.model.use_color_correction = False  # If set to True, learn transformation matrixs for appearance embedding
cfg.model.color_correction = CN()
cfg.model.color_correction.mode = 'image'  # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera senosor.
cfg.model.color_correction.use_mlp = False  # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
cfg.model.color_correction.use_sky = False  # If set to True, using spparate embedding for background and sky
# Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py

cfg.model.use_pose_correction = True  # If set to True, use pose correction for camera poses.
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image'  # If set to 'image', learn separate correction matrix for each image. If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp.
####

cfg.data = CN()
cfg.data.white_background = False  # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False  # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False  # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0  # Scale the bounding box by this factor.
cfg.data.split_test = -1
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True
cfg.data.delta_frames = 10
# data.load_pcd_from: Load the initialization point cloud from a previous experiment without generation.
# data.extent: radius of the scene, we recommend 10 - 20 meters.
# data.sphere_scale: Scale the sphere radius by this factor.
# data.regenerate_pcd: Regenerate the initialization point cloud.

cfg.render = CN()
cfg.render.antialiasing = True
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 10
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world'  # ['world', 'vehicle']
cfg.render.use_gsplat = True
cfg.render.scale = 0.01  # if not using ndc scale, this is the scale in world space, should use 0.025
cfg.render.use_ndc_scale = True
cfg.render.use_knn_scale = False
cfg.render.dist_completion = False

cfg.render.novel_view = CN()
cfg.render.novel_view.name = 'test'
cfg.render.novel_view.start_frame = -1
cfg.render.novel_view.end_frame = -1
cfg.render.novel_view.shift = [2.0, 3.0]
cfg.render.novel_view.rotate = 0.  # yaw offset
cfg.render.novel_view.steps = 10
cfg.render.novel_view.train_actor_distance_thresh = 1.5  # make sure all objects are 1 meter away from the camera during training


cfg.viewer = CN()
cfg.viewer.frame_id = 0  # Select the frame_id (start from 0) to save for viewer

# diffusion
cfg.diffusion = CN()
cfg.diffusion.use_diffusion = False
cfg.diffusion.config_path = 'video_diffusion/configs/inference/waymo_high_res.yaml'
cfg.diffusion.ckpt_path = 'video_diffusion/ckpts/vista_condition_high_res_mix_v2.safetensors'
cfg.diffusion.height = 576
cfg.diffusion.width = 1024
cfg.diffusion.save_diffusion_render = True  # easier debugging
cfg.diffusion.force_render_condition = False
cfg.diffusion.shuffle_actors = False
cfg.diffusion.masked_guidance_iter = 7000
cfg.diffusion.acc_masked_guidance = False
cfg.diffusion.cond_masked_guidance = True
cfg.diffusion.window_size = 4

cfg.profiler = CN()
cfg.profiler.enabled = False
cfg.profiler.skip_first = 5
cfg.profiler.wait = 5
cfg.profiler.warmup = 5
cfg.profiler.active = 5
cfg.profiler.repeat = 3

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(cfg, args)
