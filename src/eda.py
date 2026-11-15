"""
Dataset audit and exploratory analysis (guideline steps 3 and 5).

Produces:
  results/audit/audit.json            counts, duplicates, size stats
  results/audit/dataset_card.md       short dataset card for the report
  results/figures/eda_*.png           sample grid, size/aspect plots,
                                      foreground share per split, mean mask
"""
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

from .utils import ensure_dir, save_json, setup_plot_style, load_json

plt = setup_plot_style()


def run_audit(data, split, cfg):
    out = Path(cfg["results_dir"])
    fig_dir = ensure_dir(out / "figures")
    aud_dir = ensure_dir(out / "audit")

    n = len(data["names"])
    sizes = data["orig_sizes"]                         # (W, H)
    fg_raw = data["raw_fg"]
    fg = data["masks"].reshape(n, -1).mean(1)
    dup_groups = [h for h, c in Counter(data["hashes"].tolist()).items() if c > 1]
    meta_path = Path(cfg["data"]["cache"]).with_suffix(".meta.json")
    meta = load_json(meta_path) if meta_path.exists() else {}

    audit = {
        "n_pairs": int(n),
        "unpaired_files": meta.get("unpaired_files", []),
        "mask_raw_values_examples": meta.get("mask_raw_values_examples", []),
        "possible_duplicate_groups (avg-hash)": len(dup_groups),
        "orig_width": {"min": int(sizes[:, 0].min()), "max": int(sizes[:, 0].max()),
                       "mean": float(sizes[:, 0].mean())},
        "orig_height": {"min": int(sizes[:, 1].min()), "max": int(sizes[:, 1].max()),
                        "mean": float(sizes[:, 1].mean())},
        "aspect_ratio_W_over_H": {"min": float((sizes[:, 0] / sizes[:, 1]).min()),
                                  "max": float((sizes[:, 0] / sizes[:, 1]).max())},
        "foreground_share_256": {"mean": float(fg.mean()), "std": float(fg.std()),
                                 "min": float(fg.min()), "max": float(fg.max())},
        "foreground_share_change_after_resize_max_abs": float(np.abs(fg - fg_raw).max()),
        "empty_masks": int((fg == 0).sum()),
        "split_sizes": {k: len(split[k]) for k in ("train", "val", "test")},
        "split_foreground_mean": {k: float(fg[split[k]].mean()) for k in ("train", "val", "test")},
    }
    save_json(audit, aud_dir / "audit.json")
    _dataset_card(audit, aud_dir / "dataset_card.md")

    # ---- Figure 1: sample grid with mask overlay ----
    rng = np.random.default_rng(0)
    pick = rng.choice(n, size=min(8, n), replace=False)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.0 * len(pick), 4.3))
    for j, i in enumerate(pick):
        axes[0, j].imshow(data["images"][i]); axes[0, j].set_title(str(data["names"][i]), fontsize=8)
        axes[1, j].imshow(data["images"][i])
        axes[1, j].contour(data["masks"][i], levels=[0.5], colors="#1baf7a", linewidths=1.5)
        axes[1, j].imshow(np.ma.masked_where(data["masks"][i] == 0, data["masks"][i]),
                          cmap=ListedColormap(["#1baf7a"]), alpha=0.35)
        for a in axes[:, j]:
            a.axis("off")
    axes[0, 0].set_ylabel("image"); axes[1, 0].set_ylabel("mask overlay")
    fig.suptitle("Samples after resizing to 256x256 (bottom: ground-truth horse mask in green)", fontweight="bold")
    fig.savefig(fig_dir / "eda_samples.png"); plt.close(fig)

    # ---- Figure 2: original image sizes ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(sizes[:, 0], sizes[:, 1], s=14, color="#2a78d6", alpha=0.7)
    ax[0].set_xlabel("original width (px)"); ax[0].set_ylabel("original height (px)")
    ax[0].set_title("Original resolution")
    ax[1].hist(sizes[:, 0] / sizes[:, 1], bins=25, color="#2a78d6", edgecolor="white")
    ax[1].set_xlabel("aspect ratio W/H"); ax[1].set_ylabel("images")
    ax[1].set_title("Aspect ratio (all are resized to 1:1)")
    fig.savefig(fig_dir / "eda_sizes.png"); plt.close(fig)

    # ---- Figure 3: foreground share per split (class balance + stratification check) ----
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(0, max(0.8, fg.max()), 25)
    for k, c in (("train", "#2a78d6"), ("val", "#eb6834"), ("test", "#1baf7a")):
        ax.hist(fg[split[k]], bins=bins, density=True, histtype="step", linewidth=2, color=c,
                label=f"{k} (n={len(split[k])}, mean={fg[split[k]].mean():.2f})")
    ax.set_xlabel("fraction of pixels that are horse"); ax.set_ylabel("density")
    ax.set_title("Horse pixel share per split (stratified split)")
    ax.legend()
    fig.savefig(fig_dir / "eda_foreground_share.png"); plt.close(fig)

    # ---- Figure 4: mean training mask = spatial prior ----
    fig, ax = plt.subplots(figsize=(4, 3.8))
    im = ax.imshow(data["masks"][split["train"]].mean(0), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Mean training mask\n(where horses usually are)"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(horse)")
    fig.savefig(fig_dir / "eda_mean_mask.png"); plt.close(fig)

    # ---- Figure 5: colour of horse vs background pixels (why colour alone fails) ----
    tr = split["train"]
    imgs = data["images"][tr].reshape(-1, 3).astype(np.float32)
    msk = data["masks"][tr].reshape(-1).astype(bool)
    sub = np.random.default_rng(1).choice(len(msk), size=min(400_000, len(msk)), replace=False)
    imgs, msk = imgs[sub], msk[sub]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ci, (cname, a) in enumerate(zip(("R", "G", "B"), axes)):
        a.hist(imgs[~msk, ci], bins=64, density=True, histtype="step", color="#8a8986", label="background")
        a.hist(imgs[msk, ci], bins=64, density=True, histtype="step", color="#2a78d6", label="horse")
        a.set_title(f"{cname} channel"); a.set_xlabel("intensity")
    axes[0].set_ylabel("density"); axes[0].legend()
    fig.suptitle("Pixel colour distributions overlap heavily -> colour alone is not a reliable rule",
                 fontweight="bold")
    fig.savefig(fig_dir / "eda_colour_overlap.png"); plt.close(fig)
    print("Audit:", {k: audit[k] for k in ("n_pairs", "split_sizes", "empty_masks")})
    return audit


def _dataset_card(a, path):
    txt = f"""# Dataset card — Weizmann Horse

* Source: Borenstein & Ullman, "Class-specific, top-down segmentation", ECCV 2002.
  Kaggle mirror: https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database
* Unit of observation: one side-view RGB photograph of a horse + one binary mask (1 = horse).
* Pairs found: {a['n_pairs']} (unpaired files: {len(a['unpaired_files'])}, empty masks: {a['empty_masks']})
* Original size: width {a['orig_width']['min']}-{a['orig_width']['max']} px,
  height {a['orig_height']['min']}-{a['orig_height']['max']} px (all resized to 256x256).
* Horse pixel share (256x256): mean {a['foreground_share_256']['mean']:.3f},
  range {a['foreground_share_256']['min']:.3f}-{a['foreground_share_256']['max']:.3f}.
* Possible duplicate groups (average hash): {a['possible_duplicate_groups (avg-hash)']}
* Split (stratified by horse-size tercile, seed fixed): {a['split_sizes']}
* Mean horse share per split: { {k: round(v, 3) for k, v in a['split_foreground_mean'].items()} }
* Licence: research use; cite the original paper.
"""
    Path(path).write_text(txt)
