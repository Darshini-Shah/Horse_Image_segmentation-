"""
Tables, statistics and figures for the report. Reads only the CSV / .npy
files written by evaluate.py, so it can be re-run without a GPU.

Main outputs
  tables/main_results.csv / .md   P, R, F1, IoU (+ boundary metrics), mean ± std over seeds
  tables/significance.csv         [EXTRA] paired Wilcoxon test + bootstrap 95% CI
  figures/*.png                   all plots listed in README.md
  summary.md                      everything in one markdown file for the report
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .utils import (C_FN, C_FP, C_GT, C_PRED, COLORS, MODELS, display_name, ensure_dir,
                    load_json, setup_plot_style)

plt = setup_plot_style()

REGION = ["precision", "recall", "f1", "iou"]
BOUNDARY = ["bf1", "hd95"]
LABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1 (Dice)", "iou": "IoU",
         "bf1": "Boundary F1", "hd95": "HD95 (px)", "err_band": "near-outline error (%)",
         "err_far": "far-from-outline error (%)"}
BASE, MAIN = "baseline_noskip", "p1_concat_interp"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)


def seed_level(df):
    """Per-image metrics -> per-(model, seed) means (macro average over test images)."""
    metrics = REGION + BOUNDARY + ["err_band", "err_far"]
    g = df.groupby(["model", "seed"])
    macro = g[metrics].mean()
    # pooled ("micro") version: sum TP/FP/FN over all test pixels first
    s = g[["tp", "fp", "fn"]].sum()
    macro["iou_micro"] = s.tp / (s.tp + s.fp + s.fn)
    macro["f1_micro"] = 2 * s.tp / (2 * s.tp + s.fp + s.fn)
    return macro.reset_index()


def main_table(sl):
    order = [m for m in MODELS if m in sl.model.unique()]
    rows = []
    for m in order:
        d = sl[sl.model == m]
        r = {"Model": display_name(m), "skip": MODELS[m][1], "upsampling": MODELS[m][2]}
        for k in REGION + BOUNDARY + ["iou_micro"]:
            r[k] = f"{d[k].mean():.4f} ± {d[k].std(ddof=1) if len(d) > 1 else 0:.4f}"
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_ci(diff, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def significance(df, n_boot):
    """
    [EXTRA] Is the improvement real or noise? For every test image we average
    the metric over the seeds, then compare two models on the SAME images:
      * Wilcoxon signed-rank test (paired, no normality assumption)
      * bootstrap 95% confidence interval of the mean difference
    """
    per_img = df.groupby(["model", "test_pos"])[["iou", "f1", "bf1", "hd95"]].mean()
    pairs = [(m, BASE) for m in MODELS if m != BASE] + [("p2_add_interp", MAIN), ("p3_concat_tconv", MAIN)]
    rows = []
    for a, b in pairs:
        if a not in per_img.index.get_level_values(0) or b not in per_img.index.get_level_values(0):
            continue
        for k in ("iou", "f1", "bf1", "hd95"):
            xa, xb = per_img.loc[a][k].values, per_img.loc[b][k].values
            d = xa - xb
            try:
                p = stats.wilcoxon(xa, xb).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            lo, hi = bootstrap_ci(d, n_boot)
            better = (d < 0) if k == "hd95" else (d > 0)
            rows.append({"comparison": f"{display_name(a)} vs {display_name(b)}", "metric": LABEL[k],
                         "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
                         "wilcoxon_p": p, "images_improved_%": 100 * better.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_main_metrics(sl, path):
    order = [m for m in MODELS if m in sl.model.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3.2, 1, 1]})
    w = 0.8 / len(order)
    x = np.arange(len(REGION))
    bar_list = []
    for i, m in enumerate(order):
        d = sl[sl.model == m]
        mu, sd = d[REGION].mean().values, d[REGION].std(ddof=1).fillna(0).values
        bars = axes[0].bar(x + (i - (len(order) - 1) / 2) * w, mu, w * 0.92, yerr=sd, capsize=2,
                           color=COLORS[m], label=display_name(m),
                           error_kw={"elinewidth": 1, "ecolor": "#52514e"})
        bar_list.append((bars, mu))
    axes[0].set_xticks(x, [LABEL[k] for k in REGION])
    lo = max(0.0, sl[REGION].values.min() - 0.12)
    # value written inside the bar, near its base, so it never clashes with the error bar
    for bars, mu in bar_list:
        for b, v in zip(bars, mu):
            axes[0].text(b.get_x() + b.get_width() / 2, lo + 0.01, f"{v:.3f}", ha="center",
                         va="bottom", fontsize=7, rotation=90, color="white", fontweight="bold")
    axes[0].set_ylim(lo, 1.06)
    axes[0].set_ylabel("score (test split, mean ± std over seeds)")
    axes[0].set_title("Region metrics (higher is better)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    for ax, k, ttl in ((axes[1], "bf1", "Boundary F1 ↑"), (axes[2], "hd95", "HD95 in px ↓")):
        for i, m in enumerate(order):
            d = sl[sl.model == m][k]
            ax.bar(i, d.mean(), 0.75, yerr=d.std(ddof=1) if len(d) > 1 else 0, capsize=2, color=COLORS[m],
                   error_kw={"elinewidth": 1, "ecolor": "#52514e"})
            ax.text(i, d.mean() / 2, f"{d.mean():.2f}", ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", rotation=90)
        ax.set_xticks(range(len(order)), ["B", "P1", "P2", "P3"][:len(order)])
        ax.set_title(ttl)
    fig.suptitle("Test-set comparison of the four encoder-decoders", fontweight="bold")
    _savefig(fig, path)


def fig_training_curves(cfg, tuning, path):
    out = Path(cfg["results_dir"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for m in MODELS:
        hs = []
        for s in cfg["seeds"]:
            p = out / "histories" / f"final_{m}_{tuning['loss']}_lr{tuning['lr'][m]:g}_s{s}.csv"
            if p.exists():
                hs.append(pd.read_csv(p))
        if not hs:
            continue
        L = min(len(h) for h in hs)               # early stopping can give different lengths
        for ax, col in zip(axes, ("train_loss", "val_loss", "val_dice")):
            arr = np.stack([h[col].values[:L] for h in hs])
            ep = np.arange(1, L + 1)
            ax.plot(ep, arr.mean(0), color=COLORS[m], label=display_name(m))
            if len(hs) > 1:
                ax.fill_between(ep, arr.min(0), arr.max(0), color=COLORS[m], alpha=0.15, linewidth=0)
    for ax, t in zip(axes, ("Training loss", "Validation loss", "Validation Dice")):
        ax.set_title(t); ax.set_xlabel("epoch")
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves (line = mean over seeds, band = min-max)", fontweight="bold")
    _savefig(fig, path)


def fig_tuning(cfg, path_loss, path_lr):
    p = Path(cfg["results_dir"]) / "tuning_result.json"
    if not p.exists():
        return
    t = load_json(p)
    runs = pd.read_csv(Path(cfg["results_dir"]) / "tables" / "tuning_runs.csv")
    # loss selection
    d = runs[runs.tune_stage == "loss"]
    losses = list(dict.fromkeys(d.loss))
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [d[(d.loss == l) & (d.model == m)].best_val_dice.mean() for l in losses]
        b = ax.bar(np.arange(len(losses)) + (i - (len(models) - 1) / 2) * w, v, w * 0.92,
                   color=COLORS[m], label=display_name(m))
        ax.bar_label(b, fmt="%.3f", fontsize=7, color="#52514e")
    nice = {"bce": "BCE", "dice": "Dice", "bce_dice": "BCE + Dice"}
    ax.set_xticks(range(len(losses)), [nice.get(l, l) + (" (chosen)" if l == t["loss"] else "") for l in losses])
    vals = d.best_val_dice.values
    ax.set_ylim(max(0, vals.min() - 0.08), min(1.0, vals.max() + 0.05))
    ax.set_ylabel("best validation Dice"); ax.set_title("Loss selection on the validation split")
    ax.legend(fontsize=8)
    _savefig(fig, path_loss)
    # LR sweep
    d = runs[runs.tune_stage == "lr"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in MODELS:
        dm = d[d.model == m].sort_values("lr")
        if len(dm):
            ax.plot(dm.lr, dm.best_val_dice, "o-", color=COLORS[m], label=display_name(m), markersize=6)
    ax.set_xscale("log"); ax.set_xlabel("learning rate (log scale)"); ax.set_ylabel("best validation Dice")
    chosen = sorted(set(t["lr"].values()))
    for c in chosen:
        ax.axvline(c, color="#52514e", linestyle="--", linewidth=1)
    ax.set_title(f"Learning-rate sweep (dashed = chosen, mode: {t['lr_selection']})")
    ax.legend(fontsize=8)
    _savefig(fig, path_lr)


def fig_per_image(df, path):
    per_img = df.groupby(["model", "test_pos"])[["iou", "bf1"]].mean().reset_index()
    order = [m for m in MODELS if m in per_img.model.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    data = [per_img[per_img.model == m].iou.values for m in order]
    bp = axes[0].boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                         medianprops={"color": "#0b0b0b"})
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[m])
    rng = np.random.default_rng(0)
    for i, (d, m) in enumerate(zip(data, order)):
        axes[0].scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=COLORS[m], alpha=0.8)
    axes[0].set_xticks(range(1, len(order) + 1), ["Baseline", "P1", "P2", "P3"][:len(order)])
    axes[0].set_ylabel("IoU per test image (mean over seeds)")
    axes[0].set_title("Distribution of per-image IoU")
    for ax, k in ((axes[1], "iou"), (axes[2], "bf1")):
        if MAIN not in order:
            continue
        a = per_img[per_img.model == BASE].set_index("test_pos")[k]
        b = per_img[per_img.model == MAIN].set_index("test_pos")[k]
        ax.scatter(a, b, s=18, color=COLORS[MAIN], alpha=0.85)
        lo = min(a.min(), b.min()) - 0.02
        ax.plot([lo, 1], [lo, 1], color="#8a8986", linestyle="--", linewidth=1)
        ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
        ax.set_xlabel(f"Baseline {LABEL[k]}"); ax.set_ylabel(f"P1 {LABEL[k]}")
        ax.set_title(f"{LABEL[k]}: {100 * (b > a).mean():.0f}% of images above the diagonal")
    fig.suptitle("Per-image comparison (each dot = one test photograph)", fontweight="bold")
    _savefig(fig, path)


def _load_pred(cfg, m, seed):
    return np.load(Path(cfg["results_dir"]) / "predictions" / f"{m}_s{seed}.npy").astype(np.float32) / 255.0


def _error_rgb(img, pred, gt):
    """
    Error map: correctly found horse pixels keep their colour, everything else
    is dimmed grey; false positives orange, false negatives blue.
    """
    out = img.mean(2, keepdims=True).repeat(3, 2) / 255.0 * 0.45 + 0.5
    tp = pred & gt
    out[tp] = img[tp] / 255.0
    out[pred & ~gt] = np.array([int(C_FP[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    out[~pred & gt] = np.array([int(C_FN[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return out


def fig_outline_examples(cfg, data, split, df, path, k=2):
    """
    The problem statement asks for two test photographs where the outline is
    better. We pick them automatically: the two test images with the largest
    Boundary-F1 gain of P1 over the baseline (same seed), so the choice is
    reproducible and not hand-picked.
    """
    seed = cfg["seeds"][0]
    d = df[df.seed == seed].pivot_table(index="test_pos", columns="model", values=["bf1", "iou"])
    gain = (d["bf1"][MAIN] - d["bf1"][BASE]).sort_values(ascending=False)
    picks = gain.index[:k].tolist()
    pb, pp = _load_pred(cfg, BASE, seed), _load_pred(cfg, MAIN, seed)
    thr = cfg["eval"]["threshold"]
    fig, axes = plt.subplots(k, 5, figsize=(16, 3.4 * k))
    axes = np.atleast_2d(axes)
    for r, j in enumerate(picks):
        gi = split["test"][j]
        img, gt = data["images"][gi], data["masks"][gi].astype(bool)
        b, p = pb[j] >= thr, pp[j] >= thr
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{data['names'][gi]} (test)")
        axes[r, 1].imshow(gt, cmap="gray"); axes[r, 1].set_title("Ground-truth mask")
        for c, pred, m in ((2, b, BASE), (3, p, MAIN)):
            axes[r, c].imshow(img)
            axes[r, c].contour(gt, levels=[0.5], colors=C_GT, linewidths=1.6)
            axes[r, c].contour(pred, levels=[0.5], colors=C_PRED, linewidths=1.6)
            axes[r, c].set_title(f"{display_name(m)}\nIoU {d['iou'][m][j]:.3f} | BF1 {d['bf1'][m][j]:.3f}",
                                 fontsize=9)
        diff = _error_rgb(img, b, gt)
        # pixels the baseline got wrong but P1 fixed -> highlight in aqua
        fixed = (b ^ gt) & ~(p ^ gt)
        diff[fixed] = np.array([0.384, 0.314, 0.839])        # violet
        axes[r, 4].imshow(diff)
        axes[r, 4].set_title("Baseline errors: FP orange / FN blue\nviolet = fixed by skip connections",
                             fontsize=9)
        for a in axes[r]:
            a.axis("off")
    fig.suptitle("Outline improvement from skip connections (green = ground truth, orange = prediction)",
                 fontweight="bold")
    _savefig(fig, path)


def fig_qualitative_grid(cfg, data, split, df, path, n=6, worst=False):
    seed = cfg["seeds"][0]
    d = df[df.seed == seed].pivot_table(index="test_pos", columns="model", values="iou")
    if worst:   # failure analysis: the hardest test images for P1
        picks = d[MAIN].sort_values().index[:n].tolist()
    else:       # fixed random sample (not cherry-picked)
        picks = sorted(np.random.default_rng(0).choice(len(d), size=min(n, len(d)), replace=False).tolist())
    order = [m for m in MODELS if m in d.columns]
    preds = {m: _load_pred(cfg, m, seed) for m in order}
    cols = 2 + len(order)
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.3 * cols, 2.35 * len(picks)))
    axes = np.atleast_2d(axes)
    thr = cfg["eval"]["threshold"]
    for r, j in enumerate(picks):
        gi = split["test"][j]
        img, gt = data["images"][gi], data["masks"][gi].astype(bool)
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(gt, cmap="gray")
        if r == 0:
            axes[r, 0].set_title("Image\n", fontsize=8.5)
            axes[r, 1].set_title("Ground truth\n", fontsize=8.5)
        for c, m in enumerate(order):
            axes[r, 2 + c].imshow(_error_rgb(img, preds[m][j] >= thr, gt))
            head = f"{display_name(m)}\n" if r == 0 else ""
            axes[r, 2 + c].set_title(f"{head}IoU {d[m][j]:.3f}", fontsize=8.5)
        for a in axes[r]:
            a.axis("off")
    title = "Hardest test images for P1 (failure analysis)" if worst else "Random test images"
    fig.suptitle(f"{title} — FP orange, FN blue (seed {seed})", fontweight="bold")
    _savefig(fig, path)


def fig_band_errors(sl, path):
    order = [m for m in MODELS if m in sl.model.unique()]
    band = [sl[sl.model == m].err_band.mean() for m in order]
    far = [sl[sl.model == m].err_far.mean() for m in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(order))
    ax.barh(y, band, color=[COLORS[m] for m in order], height=0.6, label="within 3 px of the true outline")
    ax.barh(y, far, left=band, color=[COLORS[m] for m in order], alpha=0.35, height=0.6,
            label="far from the outline", edgecolor="white", linewidth=2)
    for i, (b, f) in enumerate(zip(band, far)):
        ax.text(b / 2, i, f"{b:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(b + f + 0.03, i, f"total {b + f:.2f}%", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y, [display_name(m) for m in order]); ax.invert_yaxis()
    ax.set_xlabel("misclassified pixels (% of image, mean over test images and seeds)")
    ax.set_title("Where are the errors? Near the outline (solid) vs. elsewhere (light)")
    ax.set_xlim(0, max(b + f for b, f in zip(band, far)) * 1.22)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _savefig(fig, path)


def fig_knockout(cfg, path):
    p = Path(cfg["results_dir"]) / "tables" / "skip_knockout.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    k = pd.read_csv(p)
    if k.empty:
        return None
    variants = list(dict.fromkeys(k.variant))
    models = list(dict.fromkeys(k.model))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        v = [k[(k.model == m) & (k.variant == var)].iou.mean() for var in variants]
        ax.bar(np.arange(len(variants)) + (i - (len(models) - 1) / 2) * w, v, w * 0.92,
               color=COLORS[m], label=display_name(m))
    ax.set_xticks(range(len(variants)), [v.replace("L1", "L1 (256²)").replace("L4", "L4 (32²)") for v in variants])
    ax.set_ylabel("test IoU"); ax.set_ylim(0, 1)
    ax.set_title("Skip knock-out at test time: which skip level does the trained model rely on?")
    ax.legend(fontsize=8)
    _savefig(fig, path)
    return k.groupby(["model", "variant"], sort=False).iou.mean().unstack(0)


def fig_size_bins(df, path):
    per_img = df.groupby(["model", "test_pos"]).agg(iou=("iou", "mean"), bf1=("bf1", "mean"),
                                                     fg=("fg_share", "first")).reset_index()
    q = per_img.fg.quantile([1 / 3, 2 / 3]).values
    per_img["size"] = pd.cut(per_img.fg, [-1, q[0], q[1], 2], labels=["small horse", "medium", "large horse"])
    order = [m for m in MODELS if m in per_img.model.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, k in zip(axes, ("iou", "bf1")):
        w = 0.8 / len(order)
        for i, m in enumerate(order):
            v = per_img[per_img.model == m].groupby("size", observed=False)[k].mean()
            ax.bar(np.arange(3) + (i - (len(order) - 1) / 2) * w, v.values, w * 0.92,
                   color=COLORS[m], label=display_name(m))
        ax.set_xticks(range(3), ["small horse", "medium", "large horse"])
        ax.set_ylabel(LABEL[k]); ax.set_title(f"{LABEL[k]} by horse size (test-set terciles)")
        ax.set_ylim(max(0, per_img[k].min() - 0.1), 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=8)
    _savefig(fig, path)


def fig_efficiency(eff, sl, path):
    e = eff.groupby("model").mean(numeric_only=True)
    s = sl.groupby("model").iou.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in [m for m in MODELS if m in e.index]:
        axes[0].scatter(e.loc[m, "params"] / 1e6, s[m], s=70, color=COLORS[m])
        axes[0].annotate(display_name(m), (e.loc[m, "params"] / 1e6, s[m]), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
        axes[1].scatter(e.loc[m, "infer_ms"], s[m], s=70, color=COLORS[m])
        axes[1].annotate(display_name(m), (e.loc[m, "infer_ms"], s[m]), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
    axes[0].set_xlabel("trainable parameters (millions)"); axes[0].set_ylabel("test IoU")
    axes[0].set_title("Accuracy vs. model size")
    axes[1].set_xlabel("inference time per 256x256 image (ms, batch 1)"); axes[1].set_ylabel("test IoU")
    axes[1].set_title("Accuracy vs. speed")
    for ax in axes:
        ax.margins(x=0.35, y=0.15)          # room for the text labels
    _savefig(fig, path)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def analyze(cfg, data, split, tuning):
    out = Path(cfg["results_dir"])
    tab, fig = ensure_dir(out / "tables"), ensure_dir(out / "figures")
    df = pd.read_csv(tab / "per_image_metrics.csv")
    eff = pd.read_csv(tab / "efficiency.csv")

    sl = seed_level(df)
    sl.to_csv(tab / "seed_level_metrics.csv", index=False)
    mt = main_table(sl)
    mt.to_csv(tab / "main_results.csv", index=False)
    sig = significance(df, cfg["eval"]["n_bootstrap"])
    sig.to_csv(tab / "significance.csv", index=False)
    eff_t = eff.groupby("model", sort=False).agg(
        params=("params", "first"), size_mb=("size_mb", "first"), infer_ms=("infer_ms", "mean"),
        train_time_s=("train_time_s", "mean"), best_epoch=("best_epoch", "mean"),
        val_dice=("best_val_dice", "mean")).reset_index()
    eff_t["model"] = eff_t.model.map(display_name)
    eff_t.to_csv(tab / "efficiency_summary.csv", index=False)

    fig_main_metrics(sl, fig / "results_main_metrics.png")
    fig_training_curves(cfg, tuning, fig / "training_curves.png")
    fig_tuning(cfg, fig / "tuning_loss_selection.png", fig / "tuning_lr_sweep.png")
    fig_per_image(df, fig / "results_per_image.png")
    fig_outline_examples(cfg, data, split, df, fig / "outline_two_examples.png")
    fig_qualitative_grid(cfg, data, split, df, fig / "qualitative_random.png")
    fig_qualitative_grid(cfg, data, split, df, fig / "qualitative_failures.png", n=4, worst=True)
    fig_band_errors(sl, fig / "error_near_outline.png")
    ko = fig_knockout(cfg, fig / "skip_knockout.png")
    fig_size_bins(df, fig / "results_by_horse_size.png")
    fig_efficiency(eff, sl, fig / "efficiency.png")

    # ---------- summary.md ----------
    band = sl.groupby("model", sort=False)[["err_band", "err_far"]].mean().reset_index()
    band["model"] = band.model.map(display_name)
    parts = [
        "# Results summary (auto-generated by `python main.py --stage analyze`)\n",
        f"Selected loss: **{tuning['loss']}**, learning rate(s): {tuning['lr']} "
        f"(selection mode: {tuning.get('lr_selection')}). Seeds: {cfg['seeds']}. "
        f"Threshold: {cfg['eval']['threshold']}. Test images: {len(split['test'])}.\n",
        "## Main test results (mean ± std over seeds; per-image average)\n", md_table(mt), "",
        "## Paired significance tests on the test images (seed-averaged)\n",
        md_table(sig, "{:.4g}"), "",
        "## Error location (% of image pixels)\n", md_table(band), "",
        "## Efficiency\n", md_table(eff_t, "{:.3f}"), "",
    ]
    if ko is not None:
        parts += ["## Skip knock-out (test IoU)\n", md_table(ko.reset_index(), "{:.4f}"), ""]
    tpath = out / "tuning_result.json"
    if tpath.exists():
        t = load_json(tpath)
        parts += ["## Tuning (validation Dice)\n", f"Loss table: {t['loss_table']}\n",
                  f"LR table: {t['lr_table']}\n"]
    (out / "summary.md").write_text("\n".join(parts))
    print("\n" + md_table(mt))
    print("\nSignificance:\n" + md_table(sig, "{:.4g}"))
    print(f"\nFigures in {fig}, tables in {tab}, summary in {out / 'summary.md'}")
