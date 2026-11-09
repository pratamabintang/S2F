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

from dataset_fss import fold_class_ids


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
        q = batch["query"].to(device)
        gt = batch["query_mask"].to(device)
        s_x = batch["s_x"].to(device)
        s_y = batch["s_y"].to(device)
        cls = int(batch["class_id"][0])

        out = model(q, s_x, s_y)
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
