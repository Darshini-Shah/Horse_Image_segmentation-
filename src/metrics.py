"""
Segmentation metrics, computed PER TEST IMAGE on the binarised mask
(threshold 0.5 on the sigmoid output) and then averaged over images.

Required by the problem statement:
    Precision = TP / (TP + FP)      fraction of predicted horse pixels that are horse
    Recall    = TP / (TP + FN)      fraction of true horse pixels that were found
    F1 (Dice) = 2PR / (P + R) = 2TP / (2TP + FP + FN)
    IoU       = TP / (TP + FP + FN) overlap / union

[EXTRA] Boundary metrics — the problem is about the OUTLINE, and pixel IoU
is dominated by the large interior of the horse, so it under-reports outline
quality. We therefore also measure the contour directly:
    Boundary F1 (BF score, Csurka et al. 2013): a predicted contour pixel is
    correct if a true contour pixel lies within `tol` px (and vice versa).
    HD95: 95th percentile of the symmetric distance between the two contours
    (in pixels, lower is better). Robust version of the Hausdorff distance.
[EXTRA] Boundary-band error split: wrong pixels are split into those within
`band` px of the true outline and those in the interior/background far from it.
If skip connections mainly fix the outline, the band errors should drop most.
"""
import numpy as np
from scipy import ndimage as ndi

EPS = 1e-8


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    return tp, fp, fn, tn


def region_metrics(pred, gt):
    tp, fp, fn, _ = confusion(pred, gt)
    if tp + fp + fn == 0:            # both empty -> perfect
        return dict(precision=1.0, recall=1.0, f1=1.0, iou=1.0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    return dict(precision=float(precision), recall=float(recall), f1=float(f1), iou=float(iou))


def contour(mask):
    """One-pixel-wide inner contour of a binary mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def _dist_to(mask_pts):
    """Distance map to the nearest True pixel of `mask_pts`."""
    return ndi.distance_transform_edt(~mask_pts)


def boundary_metrics(pred, gt, tol=2):
    cp, cg = contour(pred), contour(gt)
    diag = float(np.hypot(*gt.shape))
    if not cp.any() and not cg.any():
        return dict(bf1=1.0, hd95=0.0)
    if not cp.any() or not cg.any():
        return dict(bf1=0.0, hd95=diag)
    d_to_g = _dist_to(cg)[cp]        # pred contour -> nearest GT contour
    d_to_p = _dist_to(cp)[cg]        # GT contour  -> nearest pred contour
    bp = (d_to_g <= tol).mean()
    br = (d_to_p <= tol).mean()
    bf1 = 2 * bp * br / (bp + br + EPS)
    hd95 = np.percentile(np.concatenate([d_to_g, d_to_p]), 95)
    return dict(bf1=float(bf1), hd95=float(hd95))


def band_errors(pred, gt, band=3):
    """Error pixels near the true outline vs. far from it (as % of the image)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    cg = contour(gt)
    err = pred ^ gt
    near = _dist_to(cg) <= band if cg.any() else np.zeros_like(gt)
    n = gt.size
    return dict(err_band=100.0 * (err & near).sum() / n,
                err_far=100.0 * (err & ~near).sum() / n)


def all_metrics(prob, gt, thr=0.5, tol=2, band=3):
    pred = prob >= thr
    out = region_metrics(pred, gt)
    out.update(boundary_metrics(pred, gt, tol))
    out.update(band_errors(pred, gt, band))
    return out


def batch_dice_iou(pred, gt):
    """Fast torch version used for validation during training (per-image mean)."""
    import torch
    pred, gt = pred.float().flatten(1), gt.float().flatten(1)
    inter = (pred * gt).sum(1)
    ps, gs = pred.sum(1), gt.sum(1)
    dice = (2 * inter + EPS) / (ps + gs + EPS)
    iou = (inter + EPS) / (ps + gs - inter + EPS)
    return dice, iou
