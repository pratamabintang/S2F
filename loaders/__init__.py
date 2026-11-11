"""Dataset loaders for the S2F few-shot segmentation pipeline.

* `pascal_fss`    : PASCAL-5i / SBD class-episodic loader.
* `landslide_fss` : multi-modal (RGB + topography) landslide episodic loader,
                    reusing the supervised preprocessing from
                    `datasets/landslide/dataset.py`.
"""

from .pascal_fss import (
    EpisodeDataset,
    VOC_CLASSES,
    FOLD_CLASSES,
    fold_class_ids,
    read_split,
)
from .landslide_fss import LandslideEpisodeDataset

__all__ = [
    "EpisodeDataset",
    "LandslideEpisodeDataset",
    "VOC_CLASSES",
    "FOLD_CLASSES",
    "fold_class_ids",
    "read_split",
]
