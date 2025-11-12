import torch
import os
import json
import numpy as np
import imageio
from tqdm import tqdm
from street_gaussian.utils.general_utils import safe_state
from street_gaussian.config import cfg
from street_gaussian.visualizers.street_gaussian_visualizer import StreetGaussianisualizer
from street_gaussian.models.street_gaussian_model import StreetGaussianModel
from street_gaussian.models.street_gaussian_renderer import StreetGaussianRenderer
from street_gaussian.utils.diffusion_utils import getDiffusionRunner
from create_scene import create_scene
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from data_processor.waymo_processor import waymo_helpers

timer.disabled = False

def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True

    with torch.no_grad():
        scene = create_scene()
        gaussians: StreetGaussianModel = scene.gaussians
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianisualizer(save_dir)

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)
            visualizer.visualize(result, camera)

        visualizer.summarize()


def render_novel_view():
    cfg.render.save_image = False
    cfg.render.save_video = True

    with torch.no_grad():
        scene = create_scene()
        gaussians: StreetGaussianModel = scene.gaussians
        renderer = StreetGaussianRenderer()

        save_dir = os.path.join(cfg.model_path, 'novel_view', cfg.render.novel_view.name)
        novel_view_cfg = dict(cfg.render.novel_view)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(novel_view_cfg, f, indent=1)

        visualizer = StreetGaussianisualizer(save_dir)

        cameras = scene.getNovelViewCameras()
        assert cameras is not None

        novel_view_ids = list(set([camera.meta['novel_view_id'] for camera in cameras]))
        for novel_view_id in novel_view_ids:
            print(f'Rendering novel view sequence {novel_view_id}')
            cur_cameras = [camera for camera in cameras if camera.meta['novel_view_id'] == novel_view_id]
            cur_cameras = list(sorted(cur_cameras, key=lambda x: x.meta['frame']))
            for idx, camera in enumerate(tqdm(cur_cameras, desc=f"Rendering novel view sequence {novel_view_id}")):
                result = renderer.render_novel_view(camera, gaussians)
                visualizer.visualize_novel_view(result, camera)

            visualizer.result_dir = save_dir + f"/{novel_view_id}"
            os.makedirs(visualizer.result_dir, exist_ok=True)
            visualizer.summarize()
            visualizer.reset()


def run_diffusion():
    cfg.render.save_image = cfg.eval.visualize
    cfg.render.save_video = True

    with torch.no_grad():
        scene = create_scene()
        novel_cameras = scene.getNovelViewCameras()
        train_cameras = scene.getTrainCameras()

        diffusionrunner = getDiffusionRunner(scene)

        # Process novel view sequences
        if not cfg.eval.skip_novel:
            novel_view_ids = list(set([camera.meta['novel_view_id'] for camera in novel_cameras]))

            for novel_view_id in novel_view_ids:
                print(f'Running diffusion for novel view sequence {novel_view_id}')
                cur_cameras = [camera for camera in novel_cameras if camera.meta['novel_view_id'] == novel_view_id]
                cur_cameras = list(sorted(cur_cameras, key=lambda x: x.meta['frame']))

                diffusion_result = diffusionrunner.run(cur_cameras, train_cameras, use_render=False, scale=1.0)

                # Save video
                diffusion_result = (diffusion_result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                diffusion_result = np.transpose(diffusion_result, (0, 2, 3, 1))  # [num_frames, h, 3*w, 3]

                save_dir = os.path.join(cfg.model_path, 'diffusion')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'diffusion_novel_{novel_view_id}.mp4')
                imageio.mimwrite(save_path, diffusion_result, fps=10) # type: ignore

@torch.no_grad()
@catch_throw
def main():
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)

    if cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'novel_view':
        render_novel_view()
    elif cfg.mode == 'diffusion':
        run_diffusion()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    if cfg.data.get('image_widths', None) is not None:
        waymo_helpers.image_widths.clear()
        waymo_helpers.image_widths.extend(cfg.data.get('image_widths'))
    if cfg.data.get('image_heights', None) is not None:
        waymo_helpers.image_heights.clear()
        waymo_helpers.image_heights.extend(cfg.data.get('image_heights'))

    if cfg.data.get('_camera2label', None) is not None:
        waymo_helpers._camera2label.clear()
        waymo_helpers._camera2label.update(dict(cfg.data.get('_camera2label')))
    if cfg.data.get('_label2camera', None) is not None:
        _label2camera = dict(cfg.data.get('_label2camera'))
        _label2camera = {int(k[1:]): v for k,v in _label2camera.items()} # yaml中的key为_0, _1的格式，因为不能直接用整数0,1。这里转成整数
        waymo_helpers._label2camera.clear()
        waymo_helpers._label2camera.update(_label2camera)
    main()
