"""Episodic few-shot dataset for multi-modal landslide segmentation (S2F).

Reuses the EXACT preprocessing of the supervised baseline in
`datasets/landslide/dataset.py`:

  * IMAGE (optical RGB): color-jitter + CLAHE + blur photometric aug
    (train only), then ImageNet normalization
  * DTM   : per-tile min-max normalization -> [0, 1]
  * HILLSHADE : clip -> [0, 1]
  * SLOPE / ASPECT / ASPECT_COS / ASPECT_SIN : /90, /360, clip [-1, 1]
  * landslide-preserving random crop (0.8-0.95) + synchronized flips (train)
  * resize bilinear (continuous) / nearest (label)
  * `black_list.txt` filtering (broken tiles skipped)

Episode protocol: 1 query + `shot` supports. Supports ALWAYS contain landslide
pixels; the query may be positive or negative (user choice).

Directory names are auto-detected case-insensitively, so both `image/` and
`IMAGE/` layouts work.

Returns per item:
    query_rgb  : (3, H, W)          float (ImageNet-normalized)
    query_topo : (C, H, W)          float, C = len(topo_modalities)
    query_mask : (1, H, W)          float {0, 1}
    s_x_rgb    : (shot, 3, H, W)
    s_x_topo   : (shot, C, H, W)
    s_y        : (shot, H, W)       {0, 1}
    class_id   : int (1 = landslide)
    query_name : str
"""

import os
import random
import pickle
import importlib.util
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, get_worker_info
from PIL import Image

# --------------------------------------------------------------------------
# Reuse the supervised preprocessing verbatim (photometric aug, crop params,
# blacklist loader) so both pipelines stay bit-for-bit consistent.
# --------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUP_PATH = os.path.join(_REPO_ROOT, "datasets", "landslide", "dataset.py")


def _load_supervised_module():
    spec = importlib.util.spec_from_file_location(
        "landslide_supervised_dataset", _SUP_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load supervised dataset from {_SUP_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SUP = _load_supervised_module()


# --------------------------------------------------------------------------
# Path helpers (case-insensitive modality folders)
# --------------------------------------------------------------------------
DIR_CANDIDATES = {
    "IMAGE": ["image", "IMAGE", "rgb", "RGB", "optical"],
    "DTM": ["dtm", "DTM", "dem", "DEM"],
    "SLOPE": ["slope", "SLOPE"],
    "ASPECT": ["aspect", "ASPECT"],
    "ASPECT_COS": ["aspect_cos", "ASPECT_COS"],
    "ASPECT_SIN": ["aspect_sin", "ASPECT_SIN"],
    "HILLSHADE": ["hillshade", "HILLSHADE", "hill_shade"],
    "LABEL": ["label", "LABEL", "mask", "MASK", "gt", "GT", "labels"],
}

_TOPO_MODALITIES = ("DTM", "SLOPE", "ASPECT", "ASPECT_COS", "ASPECT_SIN", "HILLSHADE")


def _resolve_split_dir(data_dir: str, split: str) -> str:
    candidate = os.path.join(data_dir, split)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(data_dir):
        return data_dir
    raise FileNotFoundError(
        f"split dir not found for data_dir='{data_dir}', split='{split}'"
    )


def _resolve_modality_dir(split_dir: str, modality: str) -> str:
    for name in DIR_CANDIDATES.get(modality, [modality, modality.lower()]):
        path = os.path.join(split_dir, name)
        if os.path.isdir(path):
            return path
    available = sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    raise FileNotFoundError(
        f"no directory for modality '{modality}' in {split_dir} "
        f"(available: {available})"
    )


def _resolve_file(modality_dir: str, sample_name: str, modality: str) -> str:
    if modality == "IMAGE":
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    else:
        exts = (".tif", ".tiff", ".png", ".jpg")
    for ext in exts:
        path = os.path.join(modality_dir, sample_name + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no file for modality '{modality}' sample '{sample_name}' in {modality_dir}"
    )


class LandslideEpisodeDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "datasets/landslide",
        split: str = "train",
        size: int = 352,
        shot: int = 1,
        topo_modalities: Optional[List[str]] = None,
        n_episodes: int = 1000,
        mode: Optional[str] = None,
        blacklist_path: Optional[str] = "datasets/landslide/black_list.txt",
        cache_dir: Optional[str] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.size = size
        self.shot = shot
        self.n_episodes = n_episodes
        self.mode = mode if mode is not None else ("train" if split == "train" else "val")
        assert self.mode in ("train", "val", "test"), self.mode

        if topo_modalities is None:
            topo_modalities = ["DTM"]
        self.topo_modalities = list(topo_modalities)
        for m in self.topo_modalities:
            if m not in _TOPO_MODALITIES:
                raise ValueError(
                    f"topo modality '{m}' not supported. "
                    f"Choose from {_TOPO_MODALITIES}"
                )
        if len(self.topo_modalities) == 0:
            raise ValueError("topo_modalities must contain at least one modality")

        self.split_dir = _resolve_split_dir(data_dir, split)
        self.img_dir = _resolve_modality_dir(self.split_dir, "IMAGE")
        self.label_dir = _resolve_modality_dir(self.split_dir, "LABEL")
        self.topo_dirs = {m: _resolve_modality_dir(self.split_dir, m) for m in self.topo_modalities}

        self.blacklist = _SUP.load_blacklist(blacklist_path)

        all_names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, f))
        )
        samples, skipped = [], 0
        for name in all_names:
            if name in self.blacklist:
                skipped += 1
                continue
            samples.append(name)
        if skipped:
            print(
                f"[{self.split.upper()}] Blacklist aktif: {skipped} sampel "
                f"gambar/topografi rusak dilewati."
            )
        if not samples:
            raise RuntimeError(f"no valid samples in {self.split_dir}")
        self.samples = samples
        self.class_ids = [1]  # single foreground: landslide

        self.positive_ids = self._build_positive_index(cache_dir)
        if len(self.positive_ids) < self.shot:
            raise RuntimeError(
                f"need >= {self.shot} landslide-positive tiles in split "
                f"'{split}', found {len(self.positive_ids)}"
            )

        self.base_seed = seed
        self.epoch = 0
        self.rng = random.Random(seed)

    # ------------------------------------------------------------- indexing
    def _build_positive_index(self, cache_dir: Optional[str]) -> List[str]:
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir, f"landslide_positive_{self.split}.pkl"
            )
            if os.path.isfile(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        positives = [n for n in self.samples if bool((self._load_label(n) > 0).any())]

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(positives, f)
        return positives

    # --------------------------------------------------------- raster load
    def _load_modality(self, sample_name: str, modality: str) -> torch.Tensor:
        """Load + normalize one modality (mirrors supervised dataset.py)."""
        if modality == "IMAGE":
            path = _resolve_file(self.img_dir, sample_name, "IMAGE")
            try:
                img = Image.open(path)
                rgb_arr = np.array(img.convert("RGB"), dtype=np.uint8)
            except Exception:
                import cv2
                cv_img = cv2.imread(path)
                rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            if self.mode == "train":
                rgb_arr = _SUP.LandslideDataset._apply_photometric_augmentations(
                    None, rgb_arr
                )
            tensor = TF.to_tensor(rgb_arr)
            return TF.normalize(
                tensor,
                mean=_SUP.LandslideDataset.IMAGENET_MEAN,
                std=_SUP.LandslideDataset.IMAGENET_STD,
            )

        path = _resolve_file(self.topo_dirs[modality], sample_name, modality)
        arr = None
        try:
            with Image.open(path) as img:
                arr = np.array(img, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
        except Exception:
            pass
        if arr is None:
            import cv2
            cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cv_img is not None:
                if cv_img.ndim == 3:
                    cv_img = cv_img[:, :, 0]
                arr = cv_img.astype(np.float32)
        if arr is None:
            arr = np.zeros((self.size, self.size), dtype=np.float32)

        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if modality == "DTM":
            valid = (arr > -1000.0) & (arr < 10000.0)
            if valid.any():
                vmin, vmax = float(arr[valid].min()), float(arr[valid].max())
            else:
                vmin, vmax = 0.0, 0.0
            if vmax > vmin:
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = np.zeros_like(arr)
        elif modality == "SLOPE":
            norm = np.clip(arr / 90.0, 0.0, 1.0)
        elif modality == "ASPECT":
            norm = np.clip(arr / 360.0, 0.0, 1.0)
        elif modality in ("ASPECT_COS", "ASPECT_SIN"):
            norm = np.clip(arr, -1.0, 1.0)
        elif modality == "HILLSHADE":
            norm = np.clip(arr, 0.0, 1.0)
        else:
            raise NotImplementedError(f"loader for modality '{modality}' not implemented")

        return torch.from_numpy(np.ascontiguousarray(norm)).float().unsqueeze(0)

    def _load_label(self, sample_name: str) -> torch.Tensor:
        path = _resolve_file(self.label_dir, sample_name, "LABEL")
        try:
            arr = np.array(Image.open(path))
        except Exception:
            import cv2
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            arr = np.zeros((self.size, self.size), dtype=np.float32)
        bin_arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(bin_arr).unsqueeze(0)

    # --------------------------------------------------------- episode pick
    def __len__(self) -> int:
        return self.n_episodes

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _reseed(self, index: int):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        self.rng.seed(
            self.base_seed + self.epoch * 1_000_003 + worker * 7919 + index * 104729
        )

    def _pick_episode(self) -> Tuple[List[str], str]:
        supp_ids = self.rng.sample(self.positive_ids, self.shot)
        q_candidates = [s for s in self.samples if s not in supp_ids] or self.samples
        qry_id = self.rng.choice(q_candidates)
        return supp_ids, qry_id

    # ------------------------------------------------------------- getitem
    def _sync_transforms(
        self,
        rgb: List[torch.Tensor],
        topo: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if self.mode == "train":
            union = torch.stack(labels).amax(dim=0)  # (1, H, W)
            y0, x0, ch, cw = _SUP.LandslideDataset._get_landslide_preserving_crop_params(
                None, union, scale_range=(0.8, 0.95)
            )
            rgb = [TF.crop(t, y0, x0, ch, cw) for t in rgb]
            topo = [TF.crop(t, y0, x0, ch, cw) for t in topo]
            labels = [TF.crop(t, y0, x0, ch, cw) for t in labels]

            do_h = self.rng.random() < 0.5
            do_v = self.rng.random() < 0.5
            if do_h:
                rgb = [TF.hflip(t) for t in rgb]
                topo = [TF.hflip(t) for t in topo]
                labels = [TF.hflip(t) for t in labels]
            if do_v:
                rgb = [TF.vflip(t) for t in rgb]
                topo = [TF.vflip(t) for t in topo]
                labels = [TF.vflip(t) for t in labels]

        rgb = [
            TF.resize(t, [self.size, self.size],
                      interpolation=InterpolationMode.BILINEAR, antialias=True)
            for t in rgb
        ]
        topo = [
            TF.resize(t, [self.size, self.size],
                      interpolation=InterpolationMode.BILINEAR, antialias=True)
            for t in topo
        ]
        labels = [
            TF.resize(t, [self.size, self.size],
                      interpolation=InterpolationMode.NEAREST)
            for t in labels
        ]
        return rgb, topo, labels

    def __getitem__(self, index: int):
        self._reseed(index)
        supp_ids, qry_id = self._pick_episode()
        tiles = [qry_id] + list(supp_ids)

        rgb = [self._load_modality(t, "IMAGE") for t in tiles]
        topo = [
            torch.cat([self._load_modality(t, m) for m in self.topo_modalities], 0)
            for t in tiles
        ]
        labels = [self._load_label(t) for t in tiles]

        rgb, topo, labels = self._sync_transforms(rgb, topo, labels)

        q_rgb, s_rgb = rgb[0], torch.stack(rgb[1:], 0)
        q_topo, s_topo = topo[0], torch.stack(topo[1:], 0)
        q_mask, s_y = labels[0], torch.stack(labels[1:], 0).squeeze(1)

        return {
            "query_rgb": q_rgb,
            "query_topo": q_topo,
            "query_mask": q_mask,
            "s_x_rgb": s_rgb,
            "s_x_topo": s_topo,
            "s_y": s_y,
            "class_id": 1,
            "query_name": qry_id,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("landslide FSS loader smoke test")
    ap.add_argument("--data_dir", default="datasets/landslide")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--topo", nargs="+", default=["DTM"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--blacklist", default=None,
                    help="set to 'none' to disable blacklist")
    args = ap.parse_args()

    bl = None if (args.blacklist and args.blacklist.lower() == "none") else args.blacklist
    ds = LandslideEpisodeDataset(
        data_dir=args.data_dir, split=args.split, size=args.size, shot=args.shot,
        topo_modalities=args.topo, n_episodes=args.episodes,
        mode="train" if args.split == "train" else "val",
        blacklist_path=bl, seed=0,
    )
    print(f"samples: {len(ds.samples)}  positive: {len(ds.positive_ids)}")
    for i in range(min(args.episodes, len(ds))):
        item = ds[i]
        print(
            f"ep{i}: query={item['query_name']} "
            f"q_rgb={tuple(item['query_rgb'].shape)} "
            f"q_topo={tuple(item['query_topo'].shape)} "
            f"q_mask={tuple(item['query_mask'].shape)} "
            f"s_x_rgb={tuple(item['s_x_rgb'].shape)} "
            f"s_x_topo={tuple(item['s_x_topo'].shape)} "
            f"s_y={tuple(item['s_y'].shape)}"
        )
