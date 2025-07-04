import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from LidarPainter import LidarPintoraConvIn
from image_prep import canny_from_pil
from tqdm import tqdm

def is_gray(rgb, threshold_diff=0.05, threshold_brightness=0.4):
    """
    判断RGB像素是否为灰色（通道差异小且亮度低）
    
    参数：
    rgb: 三通道像素值（形状：[B, 3, H, W]，值域：[0, 1]）
    threshold_diff: 通道最大差值阈值（低于此值视为灰色）
    threshold_brightness: 亮度阈值（低于此值视为暗区）
    
    返回：
    gray_mask: 灰色掩码（1表示灰色，0表示非灰色）
    """
    # 计算每个像素的RGB通道最大差值
    max_val, _ = torch.max(rgb, dim=1)  # [B, H, W]
    min_val, _ = torch.min(rgb, dim=1)  # [B, H, W]
    channel_diff = max_val - min_val     # [B, H, W]
    
    # 计算亮度（使用RGB均值）
    brightness = torch.mean(rgb, dim=1)  # [B, H, W]
    
    # 判断是否为灰色（通道差异小且亮度低）
    gray_mask = (channel_diff < threshold_diff) & (brightness < threshold_brightness)
    return gray_mask.float().unsqueeze(1)  # [B, 1, H, W]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--ref_image', type=str, required=True, help='path to the reference image')
    parser.add_argument('--prompt', type=str, default='picture of urban street scene', help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    if args.model_path == '':
        raise ValueError('Model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = LidarPintoraConvIn(pretrained_path=args.model_path)
    model.set_eval()

    input_image = Image.open(args.input_image).convert('RGB')
    ref_image = Image.open(args.ref_image).convert('RGB')
    #input_image = input_image.crop((0, 166, 1600, 1066))
    #ref_image = ref_image.crop((0, 166, 1600, 1066))   #waymo数据集需要统一尺寸
    new_width = 1024; new_height = 576
    #new_width = 1664; new_height = 936
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
    # translate the image
    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
        c_ref = F.to_tensor(ref_image).unsqueeze(0).cuda()
        ref_mask = torch.any(c_ref > 0.20, dim=1, keepdim=True).float()
        #gray_mask = is_gray(c_ref)
        #ref_mask = ref_mask * (1 - gray_mask)

        lidar_region = c_ref * ref_mask
        source_region = c_t * (1 - ref_mask)
        x_combined = lidar_region + source_region

        #output_image = model(x_combined, args.prompt)
        output_image = model(c_t, x_combined, args.prompt)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        ref_pil = transforms.ToPILImage()(x_combined[0].cpu())
    # save the output image
    output_pil.save(os.path.join(args.output_dir, 'output_tgt.png'))
    ref_pil.save(os.path.join(args.output_dir, 'output_ref.png'))
