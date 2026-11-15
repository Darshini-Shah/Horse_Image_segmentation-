"""
[EXTRA] Draws the architecture diagram of the four models (for the slides /
report) straight from the same channel numbers the code uses.
"""
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .utils import COLORS, ensure_dir, setup_plot_style

plt = setup_plot_style()


def _box(ax, x, y, w, h, text, fc, ec="#52514e", fs=7.5, tc="#0b0b0b"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)


def _arrow(ax, p, q, color="#52514e", style="-|>", ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, linestyle=ls))


def draw_panel(ax, title, skip_mode, up_label, color, base=32, depth=4):
    chs = [base * 2 ** i for i in range(depth + 1)]
    res = [256 // 2 ** i for i in range(depth + 1)]
    ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.6, depth + 1.6); ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    xe, xd, w, h = 1.6, 8.4, 3.0, 0.62
    _box(ax, xe, depth + 1, 2.2, 0.5, "input 3×256²", "#f0efec")
    for i in range(depth):
        y = depth - i
        _box(ax, xe, y, w, h, f"Enc{i + 1}: {chs[i]}ch @{res[i]}²", "#dbe8f8")
        top = depth + 1 - 0.25 if i == 0 else y + 1 - h / 2      # bottom of the block above
        _arrow(ax, (xe, top), (xe, y + h / 2))
        # decoder block at same level
        _box(ax, xd, y, w, h, f"Dec{i + 1}: {chs[i]}ch @{res[i]}²", "#fde3d6")
        if skip_mode != "none":
            sym = "concat ‖" if skip_mode == "concat" else "add ⊕"
            _arrow(ax, (xe + w / 2, y), (xd - w / 2, y), color=color, ls="--", lw=1.4)
            ax.text(5, y + 0.12, sym, ha="center", va="bottom", fontsize=7.5, color=color, bbox=dict(fc="white", ec="none", pad=0.5))
    _box(ax, 5, 0, 3.2, h, f"Bottleneck {chs[-1]}ch @{res[-1]}²", "#e4e3df")
    _arrow(ax, (xe, 1 - h / 2), (5 - 1.6, 0.05))
    ax.text(0.2, 0.05, "maxpool ↓2", fontsize=7, color="#52514e")
    _arrow(ax, (5 + 1.6, 0.05), (xd, 1 - h / 2))
    ax.text(7.4, 0.05, f"{up_label} ↑2", fontsize=7, color="#52514e")
    for i in range(depth - 1):
        _arrow(ax, (xd, depth - i - 1 + h / 2), (xd, depth - i - h / 2))
    _box(ax, xd, depth + 1, 3.0, 0.5, "1×1 conv → mask 1×256²", "#f0efec")
    _arrow(ax, (xd, depth + h / 2), (xd, depth + 0.75))


def draw_architectures(out_path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    specs = [
        ("Baseline: no skip, bilinear up", "none", "bilinear+1×1", COLORS["baseline_noskip"]),
        ("P1: concat skip, bilinear up", "concat", "bilinear+1×1", COLORS["p1_concat_interp"]),
        ("P2: additive skip, bilinear up", "add", "bilinear+1×1", COLORS["p2_add_interp"]),
        ("P3: concat skip, transposed conv", "concat", "TransConv 2×2", COLORS["p3_concat_tconv"]),
    ]
    for ax, (t, s, u, c) in zip(axes, specs):
        draw_panel(ax, t, s, u, c)
    fig.text(0.5, 0.01, "Each Enc/Dec block = (3×3 conv → BatchNorm → ReLU) × 2.  Every decoder level upsamples ×2 "
             "with the listed operator; dashed arrows are skip connections.", ha="center", fontsize=9,
             color="#52514e")
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path)
    plt.close(fig)
