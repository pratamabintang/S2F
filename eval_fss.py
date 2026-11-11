"""FSS evaluation: mIoU (class-averaged) and FB-IoU.

Used by train_fss.py (validation) and test_fss.py.

Protocol (PASCAL-5i):
  * mIoU : for each fold class, accumulate intersection / union of the
           binarized query prediction vs GT over all episodes whose query
           class is that class, then average over the fold's classes.
  * FB-IoU: pool fg/bg pixels over ALL episodes, compute IoU for fg and bg,
            then average the two.
"""

import numpy as np
import torch

from loaders.pascal_fss import fold_class_ids


def unpack_batch(batch):
    """Split a PASCAL or landslide batch dict into model inputs.

    Returns (q_rgb, q_topo, q_mask, s_x_rgb, s_x_topo, s_y).
    Topography keys are absent for PASCAL (rgb-only) -> returned as None.
    """
    q_rgb = batch["query_rgb"] if "query_rgb" in batch else batch["query"]
    s_x_rgb = batch["s_x_rgb"] if "s_x_rgb" in batch else batch["s_x"]
    return (
        q_rgb,
        batch.get("query_topo"),
        batch["query_mask"],
        s_x_rgb,
        batch.get("s_x_topo"),
        batch["s_y"],
    )


class FSSMetrics:
    def __init__(self, class_ids):
        self.class_ids = list(class_ids)
        self.inter = {c: 0.0 for c in self.class_ids}
        self.union = {c: 0.0 for c in self.class_ids}
        self.fg_inter = 0.0
        self.fg_union = 0.0
        self.bg_inter = 0.0
        self.bg_union = 0.0

    @torch.no_grad()
    def add(self, pred, gt, class_id):
        # pred, gt: bool tensors (H, W) on any device
        pred = pred.bool()
        gt = gt.bool()
        tpf = (pred & gt).sum().item()
        tpb = ((~pred) & (~gt)).sum().item()
        fg_union = (pred | gt).sum().item()
        bg_union = ((~pred) | (~gt)).sum().item()

        self.inter[class_id] += tpf
        self.union[class_id] += fg_union
        self.fg_inter += tpf
        self.fg_union += fg_union
        self.bg_inter += tpb
        self.bg_union += bg_union

    def compute(self):
        eps = 1e-6
        per_class = {
            c: self.inter[c] / (self.union[c] + eps) for c in self.class_ids
        }
        miou = float(np.mean(list(per_class.values())))
        fg_iou = self.fg_inter / (self.fg_union + eps)
        bg_iou = self.bg_inter / (self.bg_union + eps)
        fbiou = float((fg_iou + bg_iou) / 2.0)
        return {"mIoU": miou, "FB-IoU": fbiou, "per_class": per_class}


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """Run episodes in `loader`; return mIoU / FB-IoU / per-class IoU."""
    model.eval()
    metrics = FSSMetrics(_class_ids_from_loader(loader))
    for batch in loader:
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

        out = model(q_rgb, q_topo, s_x_rgb, s_x_topo, s_y)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[-2:] != gt.shape[-2:]:
            out = torch.nn.functional.interpolate(
                out, size=gt.shape[-2:], mode="bilinear", align_corners=False
            )
        pred = (torch.sigmoid(out) > threshold).squeeze(0).squeeze(0)
        gt_bin = (gt > 0.5).squeeze(0).squeeze(0)
        metrics.add(pred.cpu(), gt_bin.cpu(), cls)

    return metrics.compute()


def _class_ids_from_loader(loader):
    ds = loader.dataset
    return ds.class_ids
