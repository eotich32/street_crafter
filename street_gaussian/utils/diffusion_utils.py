import os
import cv2
from typing import List
import torch
from torchvision import transforms
from PIL import Image
from street_gaussian.models.scene import Scene
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.config import cfg
from street_gaussian.utils.camera_utils import Camera
from video_diffusion.sample_condition import VideoDiffusionModel

from easyvolcap.utils.console_utils import *
from src_for_diffusion.LidarPainter import LidarPintoraConvIn
import torchvision
import torchvision.transforms.functional as F

class DiffusionRunner():
    def __init__(self, scene: Scene):
        self.scene: Scene = scene
        assert self.scene.diffusion is not None, 'Diffusion model is not found'
        assert self.scene.pointcloud_processor is not None, 'Pointcloud processor is not found'

        self.target_height = self.scene.diffusion.sample_height
        self.target_width = self.scene.diffusion.sample_width

        self.renderer = StreetGaussianRenderer()

        self.guide_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

        self.default_preprocessor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.window_size = cfg.diffusion.window_size
        self.sample_frames = self.scene.diffusion.num_frames
        print(f'Window size: {self.window_size}, sample frames: {self.sample_frames}, step: {self.sample_frames - self.window_size}')

    def get_diffusion(self):
        return self.scene.diffusion

    def get_gaussian(self):
        return self.scene.gaussians

    def get_pointcloud_processor(self):
        return self.scene.pointcloud_processor

    def get_render(self, cameras: List[Camera]):
        render_result = dict()
        render_result['render_seq'] = []
        render_result['render_mask_seq'] = []

        # for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
        for idx, camera in enumerate(cameras):
            result = self.renderer.render_novel_view(camera, self.get_gaussian())
            render_result['render_seq'].append(result['rgb'])
            render_result['render_mask_seq'].append(result['acc'])
        render_result['render_seq'] = torch.stack(render_result['render_seq'], dim=0)
        render_result['render_mask_seq'] = torch.stack(render_result['render_mask_seq'], dim=0)

        return render_result

    def get_guidance(self, cameras: List[Camera], obj_meta):
        pointcloud_processor = self.get_pointcloud_processor()
        # pointcloud_processor.render_conditions(cameras, self.scene.dataset.getmeta('obj_meta'))  # type: ignore
        pointcloud_processor.render_conditions(cameras, obj_meta)  # type: ignore
        guide_rgb_path = []
        guide_mask_path = []
        for camera in cameras:
            assert os.path.exists(camera.meta['guidance_rgb_path'])
            assert os.path.exists(camera.meta['guidance_mask_path'])
            guide_rgb_path.append(camera.meta['guidance_rgb_path'])
            guide_mask_path.append(camera.meta['guidance_mask_path'])

        return guide_rgb_path, guide_mask_path

    def preprocess_image(self, image_path, preprocessor):
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = ori_h - tmp_h
            bottom = ori_h
            # top = (ori_h - tmp_h) // 2
            # bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = preprocessor(image)
        return image

    def preprocess_tensor(self, image_tensor):
        ori_h, ori_w = image_tensor.shape[-2:]

        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image_tensor = image_tensor[..., :, left:right]
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = ori_h - tmp_h
            bottom = ori_h
            image_tensor = image_tensor[..., top:bottom, :]

        transform_resize = transforms.Resize((self.target_height, self.target_width))
        image_tensor = transform_resize(image_tensor)
        return image_tensor


class WaymoDiffusionRunner(DiffusionRunner):
    def __init__(self, scene: Scene):
        super(WaymoDiffusionRunner, self).__init__(scene)

    def run(self, cameras: List[Camera], train_cameras: List[Camera], obj_meta, use_render=True, scale: float = 0.3, masked_guidance: bool = False):
        cameras = [camera for camera in cameras if camera.meta['cam'] == 0 or camera.meta['cam'] == 1 or camera.meta['cam'] == 2]  # 3 camera
        diffusion_results = []

        novel_view_ids = list(set([camera.meta['novel_view_id'] for camera in cameras]))
        for novel_view_id in novel_view_ids:
            print(f'Running diffusion for novel view sequence {novel_view_id}')
            cur_cameras = [camera for camera in cameras if camera.meta['novel_view_id'] == novel_view_id]
            cur_cameras = list(sorted(cur_cameras, key=lambda x: x.meta['frame']))
            diffusion_result = self.run_sequence(cur_cameras, train_cameras, obj_meta, use_render, scale, masked_guidance)
            diffusion_results.append(diffusion_result)

        diffusion_results = torch.cat(diffusion_results, dim=0)
        return diffusion_results

    @torch.no_grad()
    def run_interleaved(self, test_cameras: List[Camera], train_cameras: List[Camera]):
        test_cameras = sorted(test_cameras, key=lambda x: x.meta['frame'])
        train_cameras = sorted(train_cameras, key=lambda x: x.meta['frame'])
        test_frames = [camera.meta['frame'] for camera in test_cameras]
        train_frames = [camera.meta['frame'] for camera in train_cameras]
        
        cameras = test_cameras + train_cameras
        cameras = sorted(cameras, key=lambda x: x.meta['frame'])
        frames = [camera.meta['frame'] for camera in cameras]
        num_frames = len(frames)
        sample_frames = self.sample_frames
        assert num_frames >= sample_frames, f'Not enough frames for sampling: {num_frames}'
        step = sample_frames - self.window_size
        start_idxs = list(range(0, len(frames), step))

        guide_rgb_path_all, guide_mask_path_all = self.get_guidance(cameras)

        filled = torch.zeros(num_frames, dtype=torch.bool)
        diffusion_result = torch.zeros((num_frames, 3, self.target_height, self.target_width), device='cuda')
        for start_idx in start_idxs:
            batch = dict()
            end_idx = start_idx + sample_frames
            if end_idx > num_frames:
                end_idx = num_frames
                start_idx = end_idx - sample_frames

            cond_indices = []
            guide_seq_path_sample = []
            guide_mask_seq_path_sample = []
            for idx_idx, idx in enumerate(range(start_idx, end_idx)):
                guide_seq_path_sample.append(guide_rgb_path_all[idx])
                guide_mask_seq_path_sample.append(guide_mask_path_all[idx])

            guide_seq_sample = []
            guide_mask_seq_sample = []
            for i in range(len(guide_seq_path_sample)):
                guide_seq_path = guide_seq_path_sample[i]
                guide_mask_seq_path = guide_mask_seq_path_sample[i]
                guide_seq = self.preprocess_image(guide_seq_path, self.guide_preprocessor).to('cuda', non_blocking=True)  # type: ignore
                guide_mask_seq = self.preprocess_image(guide_mask_seq_path, self.default_preprocessor).to('cuda', non_blocking=True)[..., :1, :, :]  # type: ignore
                guide_seq_sample.append(guide_seq)
                guide_mask_seq_sample.append(guide_mask_seq)
            guide_seq_sample = torch.stack(guide_seq_sample, dim=0)
            guide_mask_seq_sample = torch.stack(guide_mask_seq_sample, dim=0)

            batch['guide_seq'] = guide_seq_sample
            batch['guide_mask_seq'] = guide_mask_seq_sample

            batch['acc_masked_guidance'] = cfg.diffusion.acc_masked_guidance
            batch['cond_masked_guidance'] = cfg.diffusion.cond_masked_guidance

            img_seq = []
            img_mask_seq = []
            for idx_idx, idx in enumerate(range(start_idx, end_idx)):
                condition_camera = cameras[idx]
                cond_image = self.preprocess_tensor(condition_camera.original_image).to('cuda', non_blocking=True)
                cond_image = cond_image * 2. - 1.
                cond_image_mask = torch.ones_like(cond_image[0:1])
                img_seq.append(cond_image)
                img_mask_seq.append(cond_image_mask)
                if condition_camera.meta['frame'] in train_frames:
                    cond_indices.append(idx_idx)

            batch['img_seq'] = torch.stack(img_seq, dim=0)
            batch['img_mask_seq'] = torch.stack(img_mask_seq, dim=0)
            batch['training_free_guidance'] = False
            batch['masked_guidance'] = False
            
            print(guide_seq_path_sample)
            diffusion_output = self.scene.diffusion.forward(batch, 1.0, cond_indices=cond_indices)  # type: ignore
            diffusion_result[start_idx:end_idx] = diffusion_output  # (f, 3, h, w)
            filled[start_idx:end_idx] = True

        assert filled.all(), 'Not all frames are passed through the prior'

        for i, camera in enumerate(cameras):
            camera.meta['diffusion_original_image'] = diffusion_result[i].float().to('cuda', non_blocking=True)
            if cfg.diffusion.get('save_diffusion_render', True):
                diffusion_image = (diffusion_result[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
                diffusion_image = cv2.cvtColor(diffusion_image, cv2.COLOR_RGB2BGR)

                diffusion_save_dir = os.path.join(cfg.model_path, 'diffusion')
                os.makedirs(diffusion_save_dir, exist_ok=True)
                save_path = os.path.join(diffusion_save_dir, camera.image_name)
                save_path = save_path + '.png' if '.png' not in save_path else save_path
                cv2.imwrite(save_path, diffusion_image)

        diffusion_result = torch.stack([diffusion_result[frames.index(f)] for f in test_frames])
        return diffusion_result

    @torch.no_grad()
    def run_sequence(self, cameras: List[Camera], train_cameras: List[Camera], obj_meta, use_render=True, scale: float = 0.3, masked_guidance: bool = False):
        frames = [camera.meta['frame'] for camera in cameras]
        train_frames = np.array([camera.meta['frame'] for camera in train_cameras])

        num_frames = len(frames)
        sample_frames = self.sample_frames - 1
        assert num_frames >= sample_frames, f'Not enough frames for sampling: {num_frames}'
        step = sample_frames - self.window_size
        start_idxs = list(range(0, len(frames), step))

        guide_rgb_path_all, guide_mask_path_all = self.get_guidance(cameras, obj_meta)
        assert len(guide_rgb_path_all) == num_frames, f'Guide image should have {num_frames} frames'
        assert len(guide_mask_path_all) == num_frames, f'Guide mask should have {num_frames} frames'

        if use_render:
            render_result = self.get_render(cameras)
            render_seq_all = render_result['render_seq'].detach()  # (f, 3, h, w)
            render_mask_seq_all = render_result['render_mask_seq'].detach()  # (f, 1, h, w)
            assert render_seq_all.shape[0] == num_frames, f'Render sequence should have {num_frames} frames'
            assert render_mask_seq_all.shape[0] == num_frames, f'Render mask sequence should have {num_frames} frames'

        filled = torch.zeros(num_frames, dtype=torch.bool)
        diffusion_result = torch.zeros((num_frames, 3, self.target_height, self.target_width), device='cuda')
        for start_idx in start_idxs:
            batch = dict()
            end_idx = start_idx + sample_frames
            if end_idx > num_frames:
                end_idx = num_frames
                start_idx = end_idx - sample_frames

            start_frame = frames[start_idx]
            delta_frames = np.abs(train_frames - start_frame)
            condition_idx = np.argmin(delta_frames)
            condition_camera = train_cameras[condition_idx]
            condition_rgb_path = condition_camera.meta['guidance_rgb_path']
            condition_mask_path = condition_camera.meta['guidance_mask_path']
            guide_seq_path_sample = [condition_rgb_path] + guide_rgb_path_all[start_idx:end_idx]
            guide_mask_seq_path_sample = [condition_mask_path] + guide_mask_path_all[start_idx:end_idx]

            guide_seq_sample = []
            guide_mask_seq_sample = []
            for i in range(len(guide_seq_path_sample)):
                guide_seq_path = guide_seq_path_sample[i]
                guide_mask_seq_path = guide_mask_seq_path_sample[i]
                guide_seq = self.preprocess_image(guide_seq_path, self.guide_preprocessor).to('cuda', non_blocking=True)  # type: ignore
                guide_mask_seq = self.preprocess_image(guide_mask_seq_path, self.default_preprocessor).to('cuda', non_blocking=True)[..., :1, :, :]  # type: ignore
                guide_seq_sample.append(guide_seq)
                guide_mask_seq_sample.append(guide_mask_seq)
            guide_seq_sample = torch.stack(guide_seq_sample, dim=0)
            guide_mask_seq_sample = torch.stack(guide_mask_seq_sample, dim=0)

            batch['guide_seq'] = guide_seq_sample
            batch['guide_mask_seq'] = guide_mask_seq_sample

            batch['acc_masked_guidance'] = cfg.diffusion.acc_masked_guidance
            batch['cond_masked_guidance'] = cfg.diffusion.cond_masked_guidance

            if use_render:
                render_seq = self.preprocess_tensor(render_seq_all[start_idx:end_idx])  # type: ignore
                render_seq = render_seq * 2. - 1.
                render_mask_seq = self.preprocess_tensor(render_mask_seq_all[start_idx:end_idx])  # type: ignore
                cond_image = self.preprocess_tensor(condition_camera.original_image).to('cuda', non_blocking=True)
                cond_image = cond_image * 2. - 1.
                cond_image_mask = torch.ones_like(cond_image[0:1])
                render_seq = torch.cat([cond_image[None], render_seq], dim=0)
                render_mask_seq = torch.cat([cond_image_mask[None], render_mask_seq], dim=0)
                batch['img_seq'] = render_seq
                batch['img_mask_seq'] = render_mask_seq
                batch['training_free_guidance'] = True
                batch['masked_guidance'] = masked_guidance

            else:
                # TODO: Add disk cache for this if not using current rendering results as training free guidance
                cond_image = self.preprocess_tensor(condition_camera.original_image).to('cuda', non_blocking=True)
                cond_image = cond_image * 2. - 1.
                cond_image_mask = torch.ones_like(cond_image[0:1])
                batch['img_seq'] = torch.repeat_interleave(cond_image[None], dim=0, repeats=self.sample_frames)  # dummy img seq
                batch['img_mask_seq'] = torch.repeat_interleave(cond_image_mask[None], dim=0, repeats=self.sample_frames)  # dummy img mask seq
                batch['training_free_guidance'] = False
                batch['masked_guidance'] = False
            print(guide_seq_path_sample)
            #diffusion_output = self.scene.diffusion.forward(batch, scale, cond_indices=[0])  # type: ignore
            #diffusion_result[start_idx:end_idx] = diffusion_output[1:]  # (f, 3, h, w)
            filled[start_idx:end_idx] = True

        assert filled.all(), 'Not all frames are passed through the prior'

        for i, camera in enumerate(cameras):
            #camera.meta['diffusion_original_image'] = diffusion_result[i].float().to('cuda', non_blocking=True)
            if cfg.diffusion.get('save_diffusion_render', True) and '2.0' in camera.image_name and '-2.0' not in camera.image_name: #>>>>使用2.0用于数据处理
                #diffusion_image = (diffusion_result[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
                #diffusion_image = cv2.cvtColor(diffusion_image, cv2.COLOR_RGB2BGR)

                # Modify the guidance_rgb_path
                diffusion_save_dir = os.path.join(cfg.model_path, 'diffusion')
                os.makedirs(diffusion_save_dir, exist_ok=True)
                save_path = os.path.join(diffusion_save_dir, camera.image_name)
                save_path = save_path + '.png' if '.png' not in save_path else save_path
                #save_path = save_path.replace('.png', f'_scale{scale}.png') if scale < 1.0 else save_path
                diff_img = cv2.imread(save_path)
                #diff_img = cv2.resize(diff_img, (1024, 576), interpolation=cv2.INTER_LANCZOS4)   #>>>>数据处理尺度对齐
                diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
                image_tensor = torch.from_numpy(diff_img).permute(2, 0, 1).float() / 255.0 
                camera.meta['diffusion_original_image'] = image_tensor.to('cuda', non_blocking=True)
                #if scale == 1.0 or scale == 0.3:
                #cv2.imwrite(save_path, diffusion_image)
                # original_path = camera.meta['guidance_rgb_path']
                # modified_path = original_path.replace('color', 'prior')
                # modified_path = modified_path.replace('.png', f'_scale{scale}.png')
                # # Ensure the directory exists
                # os.makedirs(os.path.dirname(modified_path), exist_ok=True)

                # Save the diffusion image
                # cv2.imwrite(modified_path, diffusion_image)

        return diffusion_result

class ImageDiffusionRunner():
    def __init__(self, scene: Scene):
        self.scene: Scene = scene
        assert self.scene.diffusion is None, 'Should not load video diffusion model'
        assert self.scene.pointcloud_processor is not None, 'Pointcloud processor is not found'

        self.target_height = cfg.diffusion.height
        self.target_width = cfg.diffusion.width

        self.renderer = StreetGaussianRenderer()

        self.guide_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

        self.default_preprocessor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.diff_model = LidarPintoraConvIn(pretrained_path=cfg.diffusion.img_ckpt)
        self.diff_model.set_eval()
        print("Image DIffusion Model Loaded")

    def get_diffusion(self):
        return self.scene.diffusion

    def get_gaussian(self):
        return self.scene.gaussians

    def get_pointcloud_processor(self):
        return self.scene.pointcloud_processor

    def get_render(self, cameras: List[Camera]):
        render_result = dict()
        render_result['render_seq'] = []
        render_result['render_mask_seq'] = []

        # for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
        for idx, camera in enumerate(cameras):
            result = self.renderer.render_novel_view(camera, self.get_gaussian())
            render_result['render_seq'].append(result['rgb'])
            render_result['render_mask_seq'].append(result['acc'])
        render_result['render_seq'] = torch.stack(render_result['render_seq'], dim=0)
        render_result['render_mask_seq'] = torch.stack(render_result['render_mask_seq'], dim=0)

        return render_result

    def get_guidance(self, cameras: List[Camera]):
        pointcloud_processor = self.get_pointcloud_processor()
        pointcloud_processor.render_conditions(cameras, self.scene.dataset.getmeta('obj_meta'))  # type: ignore
        guide_rgb_path = []
        guide_mask_path = []
        for camera in cameras:
            assert os.path.exists(camera.meta['guidance_rgb_path'])
            assert os.path.exists(camera.meta['guidance_mask_path'])
            guide_rgb_path.append(camera.meta['guidance_rgb_path'])
            guide_mask_path.append(camera.meta['guidance_mask_path'])

        return guide_rgb_path, guide_mask_path

    def preprocess_image(self, image_path, preprocessor):
        image = Image.open(image_path)
        ori_w, ori_h = image.size
        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = ori_h - tmp_h
            bottom = ori_h
            # top = (ori_h - tmp_h) // 2
            # bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))
        image = image.resize((self.target_width, self.target_height), resample=Image.LANCZOS)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = preprocessor(image)
        return image

    def preprocess_tensor(self, image_tensor):
        ori_h, ori_w = image_tensor.shape[-2:]

        if ori_w / ori_h > self.target_width / self.target_height:
            tmp_w = int(self.target_width / self.target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image_tensor = image_tensor[..., :, left:right]
        elif ori_w / ori_h < self.target_width / self.target_height:
            tmp_h = int(self.target_height / self.target_width * ori_w)
            top = ori_h - tmp_h
            bottom = ori_h
            image_tensor = image_tensor[..., top:bottom, :]

        transform_resize = transforms.Resize((self.target_height, self.target_width))
        image_tensor = transform_resize(image_tensor)
        return image_tensor

    def run(self, cameras: List[Camera], train_cameras: List[Camera], obj_meta, use_render=True, scale: float = 0.3, masked_guidance: bool = False):
        # cameras = [camera for camera in cameras if camera.meta['cam'] == 0 or camera.meta['cam'] == 1 or camera.meta['cam'] == 2]  # 3 camera
        diffusion_results = []
        diffusion_save_dir = os.path.join(cfg.model_path, 'diffusion')
        os.makedirs(diffusion_save_dir, exist_ok=True)
        img_trans = transforms.ToPILImage()
        for camera in tqdm(cameras, desc='Running Image Diffusion'):
            #print(">>>", camera.meta['guidance_rgb_path'], camera.meta['guidance_mask_path'])
            render_result = self.renderer.render_novel_view(camera, self.get_gaussian())['rgb'].detach()
            #input_image = img_trans(render_result)
            #input_image = Image.open(args.input_image).convert('RGB')
            ref_image = Image.open(camera.meta['guidance_rgb_path']).convert('RGB')
            #input_image = input_image.crop((0, 166, 1600, 1066))
            #ref_image = ref_image.crop((0, 166, 1600, 1066))   #waymo数据集需要统一尺寸
            #new_width = 1024; new_height = 576
            #input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            #ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)

            with torch.no_grad():
                #c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
                #c_ref = F.to_tensor(ref_image).unsqueeze(0).cuda()
                c_t = self.preprocess_tensor(render_result).unsqueeze(0).cuda()
                c_ref = self.preprocess_tensor(F.to_tensor(ref_image)).unsqueeze(0).cuda()
                ref_mask = torch.any(c_ref > 0.20, dim=1, keepdim=True).float()

                lidar_region = c_ref * ref_mask
                source_region = c_t * (1 - ref_mask)
                x_combined = lidar_region + source_region

                output_image = self.diff_model(c_t, x_combined, "picture of urban street scene")
                camera.meta['diffusion_original_image'] = output_image[0]* 0.5 + 0.5
                output_pil = img_trans(output_image[0].cpu() * 0.5 + 0.5)
                ref_pil = img_trans(x_combined[0].cpu())

            save_path = os.path.join(diffusion_save_dir, camera.image_name)
            save_path = save_path + '.png' if '.png' not in save_path else save_path
            #torchvision.utils.save_image(render_result, save_path)
            output_pil.save(save_path)
            ref_pil.save(save_path.replace('.png', '_ref.png'))
            #assert "000020" not in camera.image_name, "Debug"

        return diffusion_results # empty list 

DiffusionRunnerType = {
    "Waymo": WaymoDiffusionRunner,
    "Pandaset": WaymoDiffusionRunner
}


def getDiffusionRunner(scene: Scene):
    if cfg.diffusion.use_img_diffusion:
        return ImageDiffusionRunner(scene)
    else:
        return DiffusionRunnerType[cfg.data.type](scene)
