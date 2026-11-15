"""
Creates a small synthetic "horse-like" dataset (body + neck + legs on a
noisy background) in the same folder layout as the Kaggle release.
Only used to check the pipeline end-to-end without downloading anything:

    python tools/make_dummy_dataset.py --out data/dummy --n 60
    python main.py --stage all --config configs/quick.yaml --data_root data/dummy \
        --set data.cache=data/cache_dummy.npz --results_dir results_dummy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_one(rng, W=320, H=240):
    bg = (rng.random((H, W, 3)) * 60 + rng.integers(60, 160, 3)).astype(np.uint8)
    img = Image.fromarray(bg).filter(ImageFilter.GaussianBlur(2))
    mask = Image.new("L", (W, H), 0)
    col = tuple(int(c) for c in rng.integers(40, 200, 3))
    cx, cy = rng.integers(110, 210), rng.integers(100, 140)
    bw, bh = rng.integers(60, 90), rng.integers(28, 40)
    face = rng.choice([-1, 1])
    for d in (ImageDraw.Draw(img), ImageDraw.Draw(mask)):
        fill = col if d.im.mode == "RGB" else 255
        d.ellipse([cx - bw, cy - bh, cx + bw, cy + bh], fill=fill)                       # body
        nx = cx + face * bw * 0.8
        d.polygon([(nx, cy - 10), (nx + face * 30, cy - 60), (nx + face * 45, cy - 50),
                   (nx + face * 15, cy + 5)], fill=fill)                                   # neck/head
        for lx in (cx - bw * 0.7, cx - bw * 0.4, cx + bw * 0.4, cx + bw * 0.7):
            d.rectangle([lx - 5, cy + bh * 0.6, lx + 5, cy + bh + 45], fill=fill)          # legs
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    (Path(a.out) / "horse").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "mask").mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, mask = make_one(rng)
        img.save(Path(a.out) / "horse" / f"horse{i + 1:03d}.png")
        Image.fromarray((np.array(mask) > 127).astype(np.uint8)).save(Path(a.out) / "mask" / f"horse{i + 1:03d}.png")
    print(f"wrote {a.n} pairs to {a.out}")


if __name__ == "__main__":
    main()
