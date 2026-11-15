"""
Data handling for the Weizmann Horse dataset.

Pipeline:  raw files -> pair image/mask -> resize to 256x256 -> cache (.npz)
           -> stratified train/val/test split (saved to JSON, reused by ALL models)
           -> normalisation statistics computed on TRAIN ONLY
           -> on-GPU augmentation for training batches.

The whole dataset is ~327 images, so after resizing it is only ~64 MB as
uint8. We keep it in memory (on the GPU in Colab) and index batches directly,
which avoids a DataLoader bottleneck on Colab's 2 CPU cores.
"""
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

from .utils import load_json, save_json, ensure_dir

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# folder names that hold masks in the different releases of the dataset
MASK_DIR_SUBSTR = ("mask", "figure_ground", "ground_truth")
MASK_DIR_EXACT = {"gt", "label", "labels"}
SKIP_DIR_HINTS = ("gray", "__macosx")


# ---------------------------------------------------------------------------
# 1. Find image/mask pairs
# ---------------------------------------------------------------------------
def _is_mask_path(rel: Path):
    # only look at folders BELOW the dataset root (the root path itself may contain anything)
    for part in rel.parts[:-1]:
        part = part.lower()
        if part in MASK_DIR_EXACT or any(h in part for h in MASK_DIR_SUBSTR):
            return True
    return False


def _stem_key(p: Path):
    s = p.stem.lower()
    for suf in ("_mask", "-mask", "_gt", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def discover_pairs(root):
    """
    Walk `root` and pair every RGB image with its mask by file name.
    Works for the Kaggle layout (horse/, mask/) and the original layout
    (rgb/, figure_ground/). Returns a sorted list of (img_path, mask_path).
    """
    images, masks = {}, {}
    root = Path(root)
    if not root.exists():
        return [], []
    for dirpath, _, files in os.walk(root, followlinks=True):
        rel_dir = Path(dirpath).relative_to(root)
        if any(h in str(rel_dir).lower() for h in SKIP_DIR_HINTS):
            continue
        for fn in files:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in IMG_EXT:
                continue
            (masks if _is_mask_path(rel_dir / fn) else images)[_stem_key(p)] = p
    keys = sorted(set(images) & set(masks))
    unpaired = sorted(set(images) ^ set(masks))
    return [(images[k], masks[k]) for k in keys], unpaired


# ---------------------------------------------------------------------------
# 2. Load + resize + cache
# ---------------------------------------------------------------------------
def _binarise(m: np.ndarray):
    # Kaggle masks are stored as {0,1}; some copies are {0,255}. Handle both.
    if m.ndim == 3:
        m = m[..., 0]
    thr = 0 if m.max() <= 1 else 127
    return (m > thr).astype(np.uint8)


def build_cache(cfg):
    """Read every pair once, resize to img_size x img_size and save to .npz."""
    size = cfg["data"]["img_size"]
    cache = Path(cfg["data"]["cache"])
    if cache.exists():
        return load_cache(cfg)

    pairs, unpaired = discover_pairs(cfg["data"]["root"])
    if not pairs:
        # Fallback if raw was expected but data/horse and data/mask are directly in data
        for alt in ("data", "data/raw"):
            p_alt, u_alt = discover_pairs(alt)
            if p_alt:
                pairs, unpaired = p_alt, u_alt
                break
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under {cfg['data']['root']}. "
            "See data/README.md for download instructions.")

    imgs, msks, names, orig_sizes, raw_fg, hashes, mask_values = [], [], [], [], [], [], []
    for ip, mp in pairs:
        im = Image.open(ip).convert("RGB")
        mk_raw = np.array(Image.open(mp))
        mk = _binarise(mk_raw)
        orig_sizes.append(im.size)                       # (W, H) before resizing
        raw_fg.append(mk.mean())                         # foreground share at full res
        mask_values.append(sorted(np.unique(mk_raw).tolist())[:6])
        hashes.append(_average_hash(im))
        # Problem statement: resize every image to 256 x 256.
        # Image: bilinear (smooth). Mask: nearest (keeps it strictly binary).
        im = im.resize((size, size), Image.BILINEAR)
        mk = Image.fromarray(mk * 255).resize((size, size), Image.NEAREST)
        imgs.append(np.array(im, dtype=np.uint8))
        msks.append((np.array(mk) > 127).astype(np.uint8))
        names.append(ip.stem)

    ensure_dir(cache.parent)
    np.savez_compressed(
        cache, images=np.stack(imgs), masks=np.stack(msks), names=np.array(names),
        orig_sizes=np.array(orig_sizes), raw_fg=np.array(raw_fg),
        hashes=np.array(hashes))
    save_json({"n_pairs": len(pairs), "unpaired_files": unpaired,
               "mask_raw_values_examples": mask_values[:5]},
              cache.with_suffix(".meta.json"))
    return load_cache(cfg)


def load_cache(cfg):
    d = np.load(cfg["data"]["cache"], allow_pickle=False)
    return {k: d[k] for k in d.files}


def _average_hash(im: Image.Image, n=8):
    """Tiny perceptual hash, used in the audit to look for duplicate photos."""
    g = np.array(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 3. Train / val / test split (fixed once, shared by every model)
# ---------------------------------------------------------------------------
def make_or_load_split(data, cfg, path):
    """
    With only ~327 images we use 70 / 15 / 15. The split is STRATIFIED on the
    horse size (foreground fraction tercile) so small, medium and large
    horses appear in every split. The split is saved to JSON and every model
    and every seed reads the same file -> identical evaluation protocol.
    """
    if os.path.exists(path):
        return load_json(path)
    s = cfg["data"]["split"]
    n = len(data["names"])
    idx = np.arange(n)
    fg = data["masks"].reshape(n, -1).mean(1)
    n_bins = cfg["data"].get("stratify_bins", 3)
    bins = np.digitize(fg, np.quantile(fg, np.linspace(0, 1, n_bins + 1)[1:-1]))

    seed = cfg["split_seed"]
    test_frac = s["test"]
    val_frac = s["val"] / (1.0 - test_frac)
    trval, test = train_test_split(idx, test_size=test_frac, stratify=bins, random_state=seed)
    train, val = train_test_split(trval, test_size=val_frac, stratify=bins[trval], random_state=seed)
    split = {"train": sorted(train.tolist()), "val": sorted(val.tolist()),
             "test": sorted(test.tolist()), "seed": seed,
             "names": {k: [str(data["names"][i]) for i in sorted(v.tolist())]
                       for k, v in (("train", train), ("val", val), ("test", test))}}
    # leakage check: the three index sets must be disjoint
    assert not (set(split["train"]) & set(split["val"]) or set(split["train"]) & set(split["test"])
                or set(split["val"]) & set(split["test"]))
    save_json(split, path)
    return split


def train_norm_stats(data, split):
    """Per-channel mean/std computed from TRAINING images only (no leakage)."""
    x = data["images"][split["train"]].astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# 4. Tensors on device + GPU augmentation
# ---------------------------------------------------------------------------
class SegData:
    """Holds one split as uint8 tensors on the target device."""

    def __init__(self, data, indices, device):
        self.images = torch.from_numpy(data["images"][indices]).permute(0, 3, 1, 2).contiguous().to(device)
        self.masks = torch.from_numpy(data["masks"][indices]).unsqueeze(1).contiguous().to(device)
        self.indices = list(indices)

    def __len__(self):
        return self.images.shape[0]


class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, x01):
        return (x01 - self.mean) / self.std


class GPUAugment:
    """
    Batch augmentation done directly on the GPU (fast in Colab).
    Geometric ops (flip, scale, rotation, shift) are applied identically to
    image and mask. Photometric jitter only touches the image.
    The SAME augmentation is used for all four models so the comparison is fair.
    """

    def __init__(self, cfg_aug, generator: torch.Generator):
        self.c = cfg_aug
        self.g = generator

    def _rand(self, n, lo, hi, device):
        return torch.rand(n, generator=self.g, device="cpu").to(device) * (hi - lo) + lo

    def __call__(self, x, y):
        if not self.c.get("enabled", True):
            return x, y
        B, dev = x.shape[0], x.device

        # random horizontal flip (a horse facing left is still a horse)
        if self.c.get("hflip", True):
            flip = torch.rand(B, generator=self.g) < 0.5
            flip = flip.to(dev).view(B, 1, 1, 1)
            x = torch.where(flip, x.flip(-1), x)
            y = torch.where(flip, y.flip(-1), y)

        # random affine: scale, rotation, translation via affine_grid
        lo, hi = self.c.get("scale", [1.0, 1.0])
        s = self._rand(B, lo, hi, dev)
        r = self._rand(B, -1, 1, dev) * math.radians(self.c.get("rotate_deg", 0))
        t = self.c.get("translate", 0.0)
        tx, ty = self._rand(B, -t, t, dev), self._rand(B, -t, t, dev)
        cos, sin = torch.cos(r) / s, torch.sin(r) / s
        theta = torch.stack([torch.stack([cos, -sin, tx], 1),
                             torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        # Same padding for image and mask: with "reflection" the image would get a
        # mirrored horse that the mask does not have -> wrong labels.
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # nearest for the mask so it stays binary
        y = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        # brightness / contrast / saturation jitter
        b = self.c.get("brightness", 0.0)
        c = self.c.get("contrast", 0.0)
        sat = self.c.get("saturation", 0.0)
        if b:
            x = x * self._rand(B, 1 - b, 1 + b, dev).view(B, 1, 1, 1)
        if c:
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            x = (x - m) * self._rand(B, 1 - c, 1 + c, dev).view(B, 1, 1, 1) + m
        if sat:
            gray = x.mean(1, keepdim=True)
            x = (x - gray) * self._rand(B, 1 - sat, 1 + sat, dev).view(B, 1, 1, 1) + gray
        return x.clamp(0, 1), y
