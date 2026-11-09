"""First-time dataset download + extraction for PASCAL-5i few-shot.

Archives are cached on persistent storage (`--archive_dir`, e.g. Google Drive);
extraction happens on fast LOCAL disk (`--extract_root`, e.g. /content/datasets).
This avoids writing thousands of small files to the Drive FUSE mount.

Result:
    <extract_root>/VOC2012/{JPEGImages,SegmentationClass}
    <extract_root>/SBD/{img,cls}
    <ckpt_path>  (sam2_hiera_large.pt)

Usage:
    python download_data.py \
        --archive_dir /content/drive/MyDrive/datasets/_archives \
        --extract_root /content/datasets \
        --ckpt_path /content/drive/MyDrive/checkpoints/sam2_hiera_large.pt

Existing folders/files are skipped; --overwrite forces re-download/extract.
"""

import os
import sys
import shutil
import tarfile
import argparse
import subprocess
import urllib.request


VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
# Berkeley blocks the default python-urllib UA; we send a browser UA (below).
# Keep a couple of mirror candidates for SBD.
SBD_URLS = [
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
    "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
]
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


def parse_args():
    p = argparse.ArgumentParser("download PASCAL-5i assets")
    p.add_argument("--archive_dir", type=str, required=True,
                   help="persistent dir to cache the tar/tgz archives")
    p.add_argument("--extract_root", type=str, required=True,
                   help="fast local dir to extract into (VOC2012/, SBD/)")
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="full path to save sam2_hiera_large.pt")
    p.add_argument("--voc_url", type=str, default=VOC_URL)
    p.add_argument("--sbd_url", type=str, default=None,
                   help="override SBD url (default: try built-in mirrors)")
    p.add_argument("--sam2_url", type=str, default=SAM2_URL)
    p.add_argument("--skip_voc", action="store_true")
    p.add_argument("--skip_sbd", action="store_true")
    p.add_argument("--skip_ckpt", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _stream(url, dst):
    """Download `url` to `dst` with a browser UA and a progress line."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total > 0:
                pct = 100.0 * done / total
                sys.stdout.write(
                    f"\r    {pct:5.1f}%  {done/1e6:7.1f}/{total/1e6:.1f} MB")
                sys.stdout.flush()
        sys.stdout.write("\n")


def download(url, dst, overwrite=False):
    if (not overwrite) and os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] exists: {dst}")
        return dst
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    print(f"  downloading {url}\n    -> {dst}")
    tmp = dst + ".part"
    try:
        _stream(url, tmp)
        os.replace(tmp, dst)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
    return dst


def download_any(urls, dst, overwrite=False):
    """Try each url until one succeeds."""
    last_err = None
    for url in urls:
        try:
            return download(url, dst, overwrite)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  [fail] {url}: {e}")
    raise RuntimeError(f"all mirrors failed for {dst}: {last_err}")


def extract(archive, dest):
    """Extract archive into `dest` using the `tar` CLI when available."""
    os.makedirs(dest, exist_ok=True)
    print(f"  extracting {archive} -> {dest}")
    if shutil.which("tar"):
        subprocess.run(["tar", "-xf", archive, "-C", dest], check=True)
    else:
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


def setup_voc(extract_root, archive, overwrite=False):
    target = os.path.join(extract_root, "VOC2012")
    if os.path.isdir(os.path.join(target, "JPEGImages")) and not overwrite:
        print(f"  [skip] VOC2012 ready: {target}")
        return
    tmp = os.path.join(extract_root, "_voc_tmp")
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


def setup_sbd(extract_root, archive, overwrite=False):
    target = os.path.join(extract_root, "SBD")
    if (os.path.isdir(os.path.join(target, "img"))
            and os.path.isdir(os.path.join(target, "cls")) and not overwrite):
        print(f"  [skip] SBD ready: {target}")
        return
    tmp = os.path.join(extract_root, "_sbd_tmp")
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
    os.makedirs(args.archive_dir, exist_ok=True)
    os.makedirs(args.extract_root, exist_ok=True)

    if not args.skip_voc:
        print("[VOC2012]")
        archive = download(args.voc_url,
                           os.path.join(args.archive_dir, "VOCtrainval.tar"),
                           args.overwrite)
        setup_voc(args.extract_root, archive, args.overwrite)

    if not args.skip_sbd:
        print("[SBD]")
        sbd_urls = [args.sbd_url] if args.sbd_url else SBD_URLS
        archive = download_any(sbd_urls,
                               os.path.join(args.archive_dir, "benchmark.tgz"),
                               args.overwrite)
        setup_sbd(args.extract_root, archive, args.overwrite)

    if (not args.skip_ckpt) and args.ckpt_path:
        print("[SAM2 checkpoint]")
        download(args.sam2_url, args.ckpt_path, args.overwrite)

    print("\ndone. extracted to:", args.extract_root)


if __name__ == "__main__":
    main(parse_args())
