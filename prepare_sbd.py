"""Convert SBD (Semantic Boundaries Dataset) .mat annotations to VOC-style PNG masks.

`benchmark.tgz` unpacks to:
    dataset/img/*.jpg
    dataset/cls/*.mat   (matlab: GTcls.Segmentation, pixel value = VOC class id)

This writes one PNG per image with the same pixel values (0..20, 255=void).

Usage:
    python prepare_sbd.py --cls_dir /data/SBD/cls --out_dir /data/SBD/cls_png
"""

import os
import argparse

import numpy as np
import scipy.io as sio
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser("convert SBD .mat -> .png")
    p.add_argument("--cls_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.cls_dir) if f.endswith(".mat"))
    print(f"converting {len(files)} files -> {args.out_dir}")
    for i, f in enumerate(files):
        mat = sio.loadmat(os.path.join(args.cls_dir, f))
        seg = mat["GTcls"][0]["Segmentation"][0]  # (H, W) uint8
        out = os.path.join(args.out_dir, f[:-4] + ".png")
        Image.fromarray(seg.astype(np.uint8)).save(out)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print("done")


if __name__ == "__main__":
    main(parse_args())
