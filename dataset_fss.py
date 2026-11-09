"""Episodic few-shot dataset for PASCAL-5i (1-shot protocol).

Builds episodes: pick a class present in the fold, sample `shot` support
images and 1 query image (all distinct, all containing that class), then
binarize masks (1 = the sampled class, 0 = rest).

Returns per item:
    query      : (3, H, W)  float
    query_mask : (1, H, W)  float {0,1}
    s_x        : (shot, 3, H, W)
    s_y        : (shot, 1, H, W) {0,1}
    class_id   : int (VOC class index)

Paths passed in are expected to point directly at the image / mask folders.
Masks may be VOC `SegmentationClass` PNGs (palette) or converted SBD PNGs
(pixel value = VOC class index).
"""

import os
import random
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
import torchvision.transforms.functional as TF


VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

FOLD_CLASSES = {
    0: ["aeroplane", "bicycle", "bird", "boat", "bottle"],
    1: ["bus", "car", "cat", "chair", "cow"],
    2: ["diningtable", "dog", "horse", "motorbike", "person"],
    3: ["pottedplant", "sheep", "sofa", "train", "tvmonitor"],
}

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def fold_class_ids(fold):
    """VOC class indices (1..20) belonging to a fold."""
    return [VOC_CLASSES.index(c) for c in FOLD_CLASSES[fold]]


def read_split(split_file):
    """Read an image-id list (one id per line; blank/# lines ignored)."""
    ids = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0])
    return ids


class EpisodeDataset(Dataset):
    def __init__(
        self,
        split_file,
        img_dir,
        mask_dir,
        classes,
        shot=1,
        size=352,
        mode="train",
        n_episodes=1000,
        img_ext=".jpg",
        mask_ext=".png",
        cache_dir=None,
        seed=0,
    ):
        assert mode in ("train", "val", "test")
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.class_ids = list(classes)
        self.shot = shot
        self.size = size
        self.mode = mode
        self.n_episodes = n_episodes
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.base_seed = seed
        self.epoch = 0
        self.rng = random.Random(seed)

        self.ids = read_split(split_file)
        if len(self.ids) == 0:
            raise RuntimeError(f"empty split file: {split_file}")

        self.class_to_imgs = self._build_index(cache_dir)

        # keep only classes that can form at least one full episode
        self.valid_classes = [
            c for c in self.class_ids
            if len(self.class_to_imgs.get(c, [])) >= self.shot + 1
        ]
        if not self.valid_classes:
            raise RuntimeError(
                f"no class in {classes} has >= {self.shot + 1} images "
                f"in split {split_file}"
            )

    # ---------------------------------------------------------------- index
    def _mask_path(self, imid):
        return os.path.join(self.mask_dir, imid + self.mask_ext)

    def _img_path(self, imid):
        return os.path.join(self.img_dir, imid + self.img_ext)

    def _load_mask_np(self, imid):
        arr = np.array(Image.open(self._mask_path(imid)))
        if arr.ndim == 3:  # palette saved as RGB
            arr = arr[..., 0]
        return arr

    def _build_index(self, cache_dir):
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            tag = os.path.basename(self.split_file)
            cls_tag = "-".join(str(c) for c in self.class_ids)
            cache_file = os.path.join(cache_dir, f"index_{tag}_{cls_tag}.pkl")
            if os.path.isfile(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        class_to_imgs = {c: [] for c in self.class_ids}
        for i, imid in enumerate(self.ids):
            present = set(np.unique(self._load_mask_np(imid)).tolist())
            for c in self.class_ids:
                if c in present:
                    class_to_imgs[c].append(i)

        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(class_to_imgs, f)
        return class_to_imgs

    # ---------------------------------------------------------------- items
    def __len__(self):
        return self.n_episodes

    def set_epoch(self, epoch):
        """Advance the RNG epoch so episodes differ between epochs."""
        self.epoch = epoch

    def _reseed(self, index):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        # distinct, reproducible episode per (epoch, worker, index)
        self.rng.seed(
            self.base_seed + self.epoch * 1_000_003 + worker * 7919 + index * 104729
        )

    def _pick_episode(self):
        for _ in range(50):
            c = self.rng.choice(self.valid_classes)
            pool = self.class_to_imgs[c]
            if len(pool) < self.shot + 1:
                continue
            picks = self.rng.sample(pool, self.shot + 1)
            supp_ids = [self.ids[i] for i in picks[: self.shot]]
            qry_id = self.ids[picks[self.shot]]
            return c, supp_ids, qry_id
        raise RuntimeError("failed to sample a valid episode")

    def _binarize(self, imid, c):
        m = (self._load_mask_np(imid) == c).astype(np.uint8)
        return Image.fromarray(m * 255)

    def __getitem__(self, index):
        self._reseed(index)
        c, supp_ids, qry_id = self._pick_episode()

        q_img = Image.open(self._img_path(qry_id)).convert("RGB")
        q_msk = self._binarize(qry_id, c)
        s_imgs = [Image.open(self._img_path(s)).convert("RGB") for s in supp_ids]
        s_msks = [self._binarize(s, c) for s in supp_ids]

        # resize (image bilinear, mask nearest)
        q_img = TF.resize(q_img, [self.size, self.size])
        q_msk = TF.resize(q_msk, [self.size, self.size],
                          interpolation=TF.InterpolationMode.NEAREST)
        s_imgs = [TF.resize(im, [self.size, self.size]) for im in s_imgs]
        s_msks = [TF.resize(m, [self.size, self.size],
                            interpolation=TF.InterpolationMode.NEAREST)
                  for m in s_msks]

        # consistent random flip for query + all supports (train only)
        if self.mode == "train":
            if self.rng.random() < 0.5:
                q_img = TF.hflip(q_img)
                q_msk = TF.hflip(q_msk)
                s_imgs = [TF.hflip(im) for im in s_imgs]
                s_msks = [TF.hflip(m) for m in s_msks]
            if self.rng.random() < 0.5:
                q_img = TF.vflip(q_img)
                q_msk = TF.vflip(q_msk)
                s_imgs = [TF.vflip(im) for im in s_imgs]
                s_msks = [TF.vflip(m) for m in s_msks]

        q_img = TF.normalize(TF.to_tensor(q_img), IMG_MEAN, IMG_STD)
        q_msk = (TF.to_tensor(q_msk) > 0.5).float()
        s_x = torch.stack([
            TF.normalize(TF.to_tensor(im), IMG_MEAN, IMG_STD) for im in s_imgs
        ])
        s_y = torch.stack([(TF.to_tensor(m) > 0.5).float() for m in s_msks])

        return {
            "query": q_img,
            "query_mask": q_msk,
            "s_x": s_x,
            "s_y": s_y,
            "class_id": c,
        }
