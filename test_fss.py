"""Evaluate a trained S2F checkpoint on PASCAL-5i.

Reports mIoU and FB-IoU, and optionally dumps predicted PNGs.

Example:
python test_fss.py \
  --fold 0 \
  --checkpoint runs/fold0/S2F-fold0-best.pth \
  --img_dir /data/VOC2012/JPEGImages \
  --mask_dir /data/VOC2012/SegmentationClass \
  --split /data/splits/pascal/fold0/val.txt \
  --shot 1 --size 352 --episodes 1000
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import imageio
from torch.utils.data import DataLoader

from dataset_fss import EpisodeDataset, fold_class_ids, FOLD_CLASSES, VOC_CLASSES
from eval_fss import evaluate, FSSMetrics
from models.fewshot import FewShotSeg


def parse_args():
    p = argparse.ArgumentParser("S2F few-shot test")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--mask_dir", type=str, required=True)
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--size", type=int, default=352)
    p.add_argument("--reduce_dim", type=int, default=64)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp_dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_png_dir", type=str, default=None,
                   help="if set, dump each query prediction as PNG")
    return p.parse_args()


def resolve_amp(mode, device):
    """Pick a working autocast dtype for this GPU (bf16 needs Ampere+)."""
    if mode == "none" or device.type != "cuda":
        return None
    cc_major = torch.cuda.get_device_capability()[0]
    if mode == "auto":
        mode = "bf16" if cc_major >= 8 else "fp16"
    if mode == "bf16" and cc_major < 8:
        print("[amp] GPU < Ampere -> bf16 fallback to fp16")
        mode = "fp16"
    return mode


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = fold_class_ids(args.fold)

    ds = EpisodeDataset(
        args.split, args.img_dir, args.mask_dir,
        classes=classes, shot=args.shot, size=args.size, mode="test",
        n_episodes=args.episodes, seed=args.seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    model = FewShotSeg(shot=args.shot, reduce_dim=args.reduce_dim)
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("missing keys:", missing[:5], "...")
    model.to(device).eval()

    if args.save_png_dir:
        os.makedirs(args.save_png_dir, exist_ok=True)

    amp_mode = resolve_amp(args.amp_dtype, device)
    use_amp = amp_mode is not None
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16

    metrics = FSSMetrics(classes)
    for i, batch in enumerate(loader):
        q = batch["query"].to(device)
        gt = batch["query_mask"].to(device)
        s_x = batch["s_x"].to(device)
        s_y = batch["s_y"].to(device)
        cls = int(batch["class_id"][0])

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(q, s_x, s_y)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[-2:] != gt.shape[-2:]:
            out = torch.nn.functional.interpolate(
                out, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        prob = torch.sigmoid(out)
        pred = (prob > 0.5).squeeze(0).squeeze(0)
        metrics.add(pred.cpu(), (gt > 0.5).squeeze(0).squeeze(0).cpu(), cls)

        if args.save_png_dir:
            png = (pred.cpu().numpy() * 255).astype(np.uint8)
            imageio.imsave(
                os.path.join(args.save_png_dir,
                             f"ep{i:05d}_{VOC_CLASSES[cls]}.png"), png)
        if i % 100 == 0:
            print(f"episode {i}/{args.episodes}", flush=True)

    res = metrics.compute()
    print(f"\n=== fold {args.fold} ({', '.join(FOLD_CLASSES[args.fold])}) ===")
    for c, v in res["per_class"].items():
        print(f"  IoU {VOC_CLASSES[c]:>12s}: {v:.4f}")
    print(f"  mIoU   : {res['mIoU']:.4f}")
    print(f"  FB-IoU : {res['FB-IoU']:.4f}")


if __name__ == "__main__":
    main(parse_args())
