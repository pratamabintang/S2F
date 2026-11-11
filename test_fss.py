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

Landslide:
python test_fss.py \
  --dataset landslide \
  --checkpoint runs/landslide/S2F-landslide-best.pth \
  --data_dir datasets/landslide \
  --topo DTM HILLSHADE \
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

from loaders.pascal_fss import EpisodeDataset, fold_class_ids, FOLD_CLASSES, VOC_CLASSES
from loaders.landslide_fss import LandslideEpisodeDataset
from eval_fss import evaluate, FSSMetrics, unpack_batch
from models.fewshot import FewShotSeg


def parse_args():
    p = argparse.ArgumentParser("S2F few-shot test")
    p.add_argument("--dataset", type=str, default="pascal",
                   choices=["pascal", "landslide"])
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3],
                   help="PASCAL fold (ignored for landslide)")
    p.add_argument("--checkpoint", type=str, required=True)
    # PASCAL paths
    p.add_argument("--img_dir", type=str, default=None)
    p.add_argument("--mask_dir", type=str, default=None)
    p.add_argument("--split", type=str, default=None)
    # Landslide paths
    p.add_argument("--data_dir", type=str, default="datasets/landslide")
    p.add_argument("--test_dir", type=str, default=None,
                   help="override test split dir (default: <data_dir>/test)")
    p.add_argument("--topo", nargs="+", default=["DTM"])
    p.add_argument("--blacklist", type=str,
                   default="datasets/landslide/black_list.txt",
                   help="set to 'none' to disable the blacklist")
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--size", type=int, default=352)
    p.add_argument("--reduce_dim", type=int, default=64)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp_dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "none"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=None,
                   help="cache of the per-class image index (optional)")
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

    if args.dataset == "landslide":
        bl = None if (args.blacklist and args.blacklist.lower() == "none") else args.blacklist
        classes = [1]
        ds = LandslideEpisodeDataset(
            data_dir=args.test_dir or args.data_dir, split="test",
            size=args.size, shot=args.shot,
            topo_modalities=args.topo, n_episodes=args.episodes, mode="test",
            blacklist_path=bl, cache_dir=args.cache_dir, seed=args.seed,
        )
        topo_in_chans = len(args.topo)
        split_label = f"landslide test ({args.data_dir})"
    else:
        for name in ("img_dir", "mask_dir", "split"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name} is required for --dataset pascal")
        classes = fold_class_ids(args.fold)
        ds = EpisodeDataset(
            args.split, args.img_dir, args.mask_dir,
            classes=classes, shot=args.shot, size=args.size, mode="test",
            n_episodes=args.episodes, seed=args.seed,
            cache_dir=args.cache_dir,
        )
        topo_in_chans = 1
        split_label = f"fold {args.fold} ({', '.join(FOLD_CLASSES[args.fold])})"

    def class_name(c):
        return VOC_CLASSES[c] if args.dataset == "pascal" else "landslide"

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers)

    model = FewShotSeg(shot=args.shot, reduce_dim=args.reduce_dim,
                       topo_in_chans=topo_in_chans)
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
        q_rgb, q_topo, gt, s_x_rgb, s_x_topo, s_y = unpack_batch(batch)
        q_rgb = q_rgb.to(device)
        gt = gt.to(device)
        s_x_rgb = s_x_rgb.to(device)
        s_y = s_y.to(device)
        if q_topo is not None:
            q_topo = q_topo.to(device)
        if s_x_topo is not None:
            s_x_topo = s_x_topo.to(device)
        cls = int(batch["class_id"][0])

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(q_rgb, q_topo, s_x_rgb, s_x_topo, s_y)
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
                             f"ep{i:05d}_{class_name(cls)}.png"), png)
        if i % 100 == 0:
            print(f"episode {i}/{args.episodes}", flush=True)

    res = metrics.compute()
    print(f"\n=== {split_label} ===")
    for c, v in res["per_class"].items():
        print(f"  IoU {class_name(c):>12s}: {v:.4f}")
    print(f"  mIoU   : {res['mIoU']:.4f}")
    print(f"  FB-IoU : {res['FB-IoU']:.4f}")


if __name__ == "__main__":
    main(parse_args())
