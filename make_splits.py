"""Build PASCAL-5i fold split files by scanning masks for fold classes.

Use this if you do NOT have the official FSS split lists. It creates
`{out_dir}/fold{i}/train.txt` (from SBD) and `val.txt` (from VOC2012 val),
where each file lists the image ids whose mask contains >=1 class of fold i.

Note: for numbers comparable to the literature, prefer the official split
lists; this generator is a self-contained fallback.

Usage:
    python make_splits.py \
        --sbd_img_dir /data/SBD/img --sbd_mask_dir /data/SBD/cls_png \
        --voc_img_dir /data/VOC2012/JPEGImages \
        --voc_mask_dir /data/VOC2012/SegmentationClass \
        --out_dir /data/splits/pascal
"""

import os
import argparse

import numpy as np
from PIL import Image

from dataset_fss import FOLD_CLASSES, VOC_CLASSES


def parse_args():
    p = argparse.ArgumentParser("build PASCAL-5i splits")
    p.add_argument("--sbd_img_dir", type=str, required=True)
    p.add_argument("--sbd_mask_dir", type=str, required=True)
    p.add_argument("--voc_img_dir", type=str, required=True)
    p.add_argument("--voc_mask_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def _mask_classes(mask_path):
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return set(np.unique(arr).tolist())


def _collect(img_dir, mask_dir, class_ids):
    ids = []
    names = [f for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for i, f in enumerate(sorted(names)):
        imid = os.path.splitext(f)[0]
        mp = os.path.join(mask_dir, imid + ".png")
        if not os.path.isfile(mp):
            continue
        if _mask_classes(mp) & set(class_ids):
            ids.append(imid)
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(names)} in {img_dir}", flush=True)
    return ids


def main(args):
    for fold in range(4):
        class_ids = [VOC_CLASSES.index(c) for c in FOLD_CLASSES[fold]]
        fold_dir = os.path.join(args.out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"[fold {fold}] scanning SBD ...")
        train_ids = _collect(args.sbd_img_dir, args.sbd_mask_dir, class_ids)
        print(f"[fold {fold}] scanning VOC2012 ...")
        val_ids = _collect(args.voc_img_dir, args.voc_mask_dir, class_ids)

        with open(os.path.join(fold_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_ids) + "\n")
        with open(os.path.join(fold_dir, "val.txt"), "w") as f:
            f.write("\n".join(val_ids) + "\n")
        print(f"[fold {fold}] train={len(train_ids)} val={len(val_ids)}")


if __name__ == "__main__":
    main(parse_args())
