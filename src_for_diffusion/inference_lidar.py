import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from LidarPainter import LidarPainter
from image_prep import canny_from_pil
from tqdm import tqdm

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
    model = LidarPainter(pretrained_path=args.model_path)
    model.set_eval()

    input_image = Image.open(args.input_image).convert('RGB')
    ref_image = Image.open(args.ref_image).convert('RGB')
    new_width = 1024; new_height = 576
    #new_width = 1664; new_height = 936
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
    # translate the image
    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
        c_ref = F.to_tensor(ref_image).unsqueeze(0).cuda()

        output_image, output_ref = model(c_t, c_ref, args.prompt)
        #output_image = model(c_t, c_ref, args.prompt)
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        ref_pil = transforms.ToPILImage()(output_ref[0].cpu() * 0.5 + 0.5)
    # save the output image
    output_pil.save(os.path.join(args.output_dir, 'output_tgt.png'))
    ref_pil.save(os.path.join(args.output_dir, 'output_ref.png'))
