"""First-time dataset download + extraction for PASCAL-5i few-shot.

Downloads (if missing):
  * PASCAL VOC2012 devkit  -> <root>/VOC2012/{JPEGImages,SegmentationClass}
  * SBD benchmark.tgz      -> <root>/SBD/{img,cls}
  * SAM2 Hiera-L ckpt      -> <ckpt_path>

Usage:
    python download_data.py --root /content/drive/MyDrive/datasets/PASCAL \
        --ckpt_path /content/drive/MyDrive/checkpoints/sam2_hiera_large.pt

By default an existing folder/file is not re-downloaded. Use --overwrite to force.
"""

import os
import sys
import shutil
import tarfile
import argparse
import urllib.request


VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
SBD_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


def parse_args():
    p = argparse.ArgumentParser("download PASCAL-5i assets")
    p.add_argument("--root", type=str, required=True,
                   help="dir that will contain VOC2012/ and SBD/")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="full path to save sam2_hiera_large.pt")
    p.add_argument("--voc_url", type=str, default=VOC_URL)
    p.add_argument("--sbd_url", type=str, default=SBD_URL)
    p.add_argument("--sam2_url", type=str, default=SAM2_URL)
    p.add_argument("--skip_voc", action="store_true")
    p.add_argument("--skip_sbd", action="store_true")
    p.add_argument("--skip_ckpt", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    done = min(count * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r    {pct:5.1f}%  {done/1e6:7.1f}/{total_size/1e6:.1f} MB")
    sys.stdout.flush()
    if done >= total_size:
        sys.stdout.write("\n")


def download(url, dst, overwrite=False):
    if (not overwrite) and os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] exists: {dst}")
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    print(f"  downloading {url}\n    -> {dst}")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    os.replace(tmp, dst)
    return dst


def extract(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"  extracting {archive} -> {dest}")
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # python < 3.12
            tar.extractall(dest)


def _find_dir(base, marker):
    for root, dirs, _ in os.walk(base):
        if marker in dirs:
            return root
    return None


def setup_voc(root, archive, overwrite=False):
    target = os.path.join(root, "VOC2012")
    if os.path.isdir(os.path.join(target, "JPEGImages")) and not overwrite:
        print(f"  [skip] VOC2012 ready: {target}")
        return
    tmp = os.path.join(root, "_voc_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    extract(archive, tmp)
    src = _find_dir(tmp, "JPEGImages")
    if src is None:
        raise RuntimeError("VOC2012/JPEGImages not found after extraction")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.move(src, target)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  VOC2012 ready: {target}")


def setup_sbd(root, archive, overwrite=False):
    target = os.path.join(root, "SBD")
    if (os.path.isdir(os.path.join(target, "img"))
            and os.path.isdir(os.path.join(target, "cls")) and not overwrite):
        print(f"  [skip] SBD ready: {target}")
        return
    tmp = os.path.join(root, "_sbd_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    extract(archive, tmp)
    src = _find_dir(tmp, "img")
    if src is None:
        raise RuntimeError("SBD img/ not found after extraction")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.move(src, target)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  SBD ready: {target}")


def main(args):
    os.makedirs(args.root, exist_ok=True)

    if not args.skip_voc:
        print("[VOC2012]")
        archive = download(args.voc_url, os.path.join(args.root, "VOCtrainval.tar"),
                           args.overwrite)
        setup_voc(args.root, archive, args.overwrite)

    if not args.skip_sbd:
        print("[SBD]")
        archive = download(args.sbd_url, os.path.join(args.root, "benchmark.tgz"),
                           args.overwrite)
        setup_sbd(args.root, archive, args.overwrite)

    if (not args.skip_ckpt) and args.ckpt_path:
        print("[SAM2 checkpoint]")
        download(args.sam2_url, args.ckpt_path, args.overwrite)

    print("\ndone.")


if __name__ == "__main__":
    main(parse_args())
