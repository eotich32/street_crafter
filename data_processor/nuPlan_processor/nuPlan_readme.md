**nuPlan数据预处理**

**1.数据结构**

![](https://pingcode.51aes.com/atlas/files/public/68ec987aad5473847dbfcbd1?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWRfZm9yX3B1YmljX2ltYWdlIjoiOGI2YmY1ZjE0MGIzNDA5M2I3ODIxMGE4ODVkZjI0ZmUiLCJ0ZWFtX2Zvcl9wdWJsaWNfaW1hZ2UiOiI2NTYxNTM3ODJjNTk1MGIxZjRhMGIwZmQiLCJpc19pbnRlcm5hbF90b2tlbiI6dHJ1ZSwiaWF0IjoxNzYxMTAyNDM4LCJleHAiOjE3NjExMTMyMzh9.19SsR85RhuW_EwXE2Q4oL06uE9xJ0Tnx8cAZ7WzeWaQ)

需要严格按照官网的结构进行数据集组织

**2.nuplan_converter.py**

转换可以得到dynamic_mask,images,ego_poses,extrinsics,intrinsics,track,vis_videos等，

conda activate streetcrafter
python data_processor/nuPlan_processor/nuplan_converter.py \
--nuplan_root /mnt/data/dataset/nuPlan/raw 
--log_name 2021.05.12.22.00.38_veh-35_01008_01518 \
--save_dir /mnt/data/dataset/nuPlan/processed/01518_frames_0_200 \
--start_frame 0 --num_frames 200 \
--cam_ids 0 1 2 3 4 5 6 7

其中log_name是进行不同场景的选择，frame_num是进行选取多少帧，cam_ids是选择多少个相机。

**3.nuplan_point_converter.py**

转换可以得到background,actor点云，还有深度depth
conda activate streetcrafter
python data_processor/nuPlan_processor/nuplan_get_lidar_pcd.py \
--nuplan_root /mnt/data/dataset/nuPlan/raw \
--log_name 2021.05.12.22.00.38_veh-35_01008_01518 \
--save_dir /mnt/data/dataset/nuPlan/processed/01518_frames_0_200 \

其中log_name依旧是进行不同场景的选择

**4.generate_sky_mask_with_8cams.py**
conda activate sky_mask_generate
python  data_processor/nuPlan_processor/generate_sky_mask_with_8cams.py \
--datadir /mnt/data/dataset/nuPlan/processed/2021.05.12.22.00.38_veh-35_01008_01518 \
--sam_checkpoint /src/51sim-ai/street_crafter_copy/sky_checkpoint/sam_vit_h_4b8939.pth

其中--datadir是处理后的数据目录

**5.nuplan_render_lidar_pcd.py**

这个文件在generate_sky_mask_with_8cams.py同目录下，是调试雷达点云和actor点云以及相机位姿是否可以正确投影的测试脚本，可以不使用。

**数据加载**

数据加载部分需要修改street_gaussian/datasets/waymo_readers.py和/src/51sim-ai/street_crafter_copy/data_processor/waymo_processor/waymo_helpers.py文件下的内容，具体修改是waymo_helper.py中的开头的image_heights和image_widths，以及_camera2label和_label2camera，以及往后所有的循环，将range(5)改成range(8）。waymo_reader只需要把48行的改成'cameras=cfg.data.get('cameras', [0, 1, 2, 3, 4, 5, 6, 7])

**训练过程**

将train函数中的输出log参数加上点云大小progress_bar.set_description(f"Exp: {cfg.task}-{cfg.exp_name}, Loss: {ema_loss_for_log:.{7}f}, PSNR: {ema_psnr_for_log:.{4}f}, point_num: {gaussians.get_xyz.size(0)}")。经过测试，点云point_num在40000000个的时候就无法再继续进行训练了。

需要修改config中的参数，主要有densify_grad_threshold_bkgd和densify_grad_abs_obj，这个是在密度化的角度来实现，还有从剪枝的阈值进行的是min_opacity，第一个已经经过测试是有效的，第二个还需进行测试。

**出现的问题**

![](C:\Users\lvyexiangzi\Desktop\9020.jpg)

如果出现下方动态物体出现重影，这个大概率是ego_pose中的frame id_cam id的文件出现了问题，需要进行检查

![10410.jpg](C:\Users\lvyexiangzi\Desktop\10410.jpg)

还存在着这种明显有车，并且明显存在跟踪信息的

![](C:\Users\lvyexiangzi\AppData\Roaming\marktext\images\2025-10-22-11-45-08-image.png)

这大概率是和dynamic_mask有关系，因为车已经停了，所以判断成静态背景了。

![](C:\Users\lvyexiangzi\Desktop\000090_1.png)
