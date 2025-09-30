#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse

def overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    """mask: 0/255 单通道"""
    overlay = image.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def process_one(img_path, mask_path, out_path, color, alpha):
    img  = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"[WARN] skip {img_path}")
        return
    out = overlay(img, mask, color=color, alpha=alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"[SAVE] {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",    type=str, help="单帧图像")
    parser.add_argument("--mask",   type=str, help="单帧掩码")
    parser.add_argument("--out",    type=str, help="单帧输出")
    parser.add_argument("--seq_dir",type=str, help="整序列根目录（自动配对）")
    parser.add_argument("--out_dir",type=str, help="整序列输出目录")
    parser.add_argument("--color", type=int, nargs=3, default=[0, 0, 255], help="B,G,R")
    parser.add_argument("--alpha", type=float, default=0.5, help="不透明度 0~1")
    args = parser.parse_args()

    color = tuple(args.color)

    # 单帧模式
    if args.img and args.mask and args.out:
        process_one(args.img, args.mask, Path(args.out), color, args.alpha)
        return

    # 整序列模式
    if not args.seq_dir or not args.out_dir:
        print("[ERROR] 请提供 --seq_dir 和 --out_dir")
        return

    seq_dir  = Path(args.seq_dir)
    out_dir  = Path(args.out_dir)
    img_dir  = seq_dir / "images"
    mask_dir = seq_dir / "dynamic_mask"

    for img_file in sorted(img_dir.glob("*.jpg")):
        mask_file = mask_dir / img_file.name
        if not mask_file.exists():
            continue
        out_file = out_dir / img_file.name
        process_one(img_file, mask_file, out_file, color, args.alpha)

if __name__ == "__main__":
    main()