"""Episodic training for few-shot segmentation on PASCAL-5i (1-shot default).

Trains one fold at a time (5 classes) and saves checkpoints to --save_path.

Example:
python train_fss.py \
  --fold 0 \
  --hiera_path /path/sam2_hiera_large.pt \
  --train_img_dir /data/SBD/img \
  --train_mask_dir /data/SBD/cls_png \
  --train_split /data/splits/pascal/fold0/train.txt \
  --val_img_dir /data/VOC2012/JPEGImages \
  --val_mask_dir /data/VOC2012/SegmentationClass \
  --val_split /data/splits/pascal/fold0/val.txt \
  --save_path runs/fold0 \
  --shot 1 --size 352
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_fss import EpisodeDataset, fold_class_ids
from eval_fss import evaluate
from models.fewshot import FewShotSeg


def resolve_amp(mode, device):
    """Pick a working autocast dtype for this GPU (bf16 needs Ampere+)."""
    if mode == "none" or device.type != "cuda":
        return None
    cc_major = torch.cuda.get_device_capability()[0]
    if mode == "auto":
        mode = "bf16" if cc_major >= 8 else "fp16"
    if mode == "bf16" and cc_major < 8:
        print("[amp] GPU < Ampere (sm_%d) -> bf16 fallback to fp16" % cc_major,
              flush=True)
        mode = "fp16"
    return mode


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def parse_args():
    p = argparse.ArgumentParser("S2F few-shot")
    p.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3])
    p.add_argument("--hiera_path", type=str, required=True)
    p.add_argument("--train_img_dir", type=str, required=True)
    p.add_argument("--train_mask_dir", type=str, required=True)
    p.add_argument("--train_split", type=str, required=True)
    p.add_argument("--val_img_dir", type=str, required=True)
    p.add_argument("--val_mask_dir", type=str, required=True)
    p.add_argument("--val_split", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--size", type=int, default=352)
    p.add_argument("--reduce_dim", type=int, default=64)
    # episodes
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--episodes_per_epoch", type=int, default=1000)
    p.add_argument("--val_episodes", type=int, default=1000)
    p.add_argument("--val_interval", type=int, default=1)
    # optimization
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp_dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "none"],
                   help="mixed precision: auto (bf16 on Ampere+, fp16 else), or forced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=None,
                   help="cache of the per-class image index (optional)")
    return p.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = fold_class_ids(args.fold)

    train_ds = EpisodeDataset(
        args.train_split, args.train_img_dir, args.train_mask_dir,
        classes=classes, shot=args.shot, size=args.size, mode="train",
        n_episodes=args.episodes_per_epoch, cache_dir=args.cache_dir,
        seed=args.seed,
    )
    val_ds = EpisodeDataset(
        args.val_split, args.val_img_dir, args.val_mask_dir,
        classes=classes, shot=args.shot, size=args.size, mode="val",
        n_episodes=args.val_episodes, cache_dir=args.cache_dir,
        seed=args.seed + 12345,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    model = FewShotSeg(shot=args.shot, reduce_dim=args.reduce_dim,
                       checkpoint_path=args.hiera_path)
    model.to(device)

    optim = opt.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epochs, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    amp_mode = resolve_amp(args.amp_dtype, device)
    use_amp = amp_mode is not None
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    # bf16 needs no gradient scaling; fp16 does
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_mode == "fp16"))
    print(f"AMP: {amp_mode or 'none'} (enabled={use_amp})", flush=True)

    best_miou = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        running = 0.0
        for i, batch in enumerate(train_loader):
            q = batch["query"].to(device)
            target = batch["query_mask"].to(device)
            s_x = batch["s_x"].to(device)
            s_y = batch["s_y"].to(device)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                out, out1, out2 = model(q, s_x, s_y)
                loss = (structure_loss(out, target)
                        + structure_loss(out1, target)
                        + structure_loss(out2, target))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()
        scheduler.step()
        avg = running / max(1, len(train_loader))

        if (epoch + 1) % args.val_interval == 0:
            metrics = evaluate(model, val_loader, device)
            print(f"[fold {args.fold}] epoch {epoch+1} train_loss {avg:.4f} "
                  f"mIoU {metrics['mIoU']:.4f} FB-IoU {metrics['FB-IoU']:.4f}",
                  flush=True)
            ckpt = os.path.join(args.save_path, f"S2F-fold{args.fold}-last.pth")
            torch.save(model.state_dict(), ckpt)
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                torch.save(model.state_dict(),
                           os.path.join(args.save_path,
                                        f"S2F-fold{args.fold}-best.pth"))
                print(f"[fold {args.fold}] new best mIoU {best_miou:.4f}",
                      flush=True)
        else:
            print(f"[fold {args.fold}] epoch {epoch+1} train_loss {avg:.4f}",
                  flush=True)

    print(f"[fold {args.fold}] done. best mIoU {best_miou:.4f}")


if __name__ == "__main__":
    main(parse_args())
