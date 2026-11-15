"""
Small helpers shared by every stage: seeding, device, config loading,
JSON/CSV logging and the plotting style used for all figures.
"""
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Model registry. Every model is the SAME encoder-decoder; only the skip
# connection type and the decoder upsampling operator change.
# key -> (display name, skip_mode, up_mode)
# ---------------------------------------------------------------------------
MODELS = {
    "baseline_noskip":  ("Baseline (no skip)",      "none",   "interp"),
    "p1_concat_interp": ("P1: Concat skip",         "concat", "interp"),
    "p2_add_interp":    ("P2: Additive skip",       "add",    "interp"),
    "p3_concat_tconv":  ("P3: Concat + TransConv",  "concat", "tconv"),
}

# Fixed colour per model (colour follows the model in every figure).
# Baseline is neutral grey so the proposed models stand out against it.
COLORS = {
    "baseline_noskip":  "#8a8986",
    "p1_concat_interp": "#2a78d6",
    "p2_add_interp":    "#eb6834",
    "p3_concat_tconv":  "#1baf7a",
}

# Colours used in overlays (GT vs prediction, FP vs FN)
C_GT, C_PRED = "#1baf7a", "#eb6834"
C_FP, C_FN = "#eb6834", "#2a78d6"


def display_name(key):
    return MODELS[key][0]


def set_seed(seed: int):
    """Fix every RNG we use so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: pick deterministic kernels (slightly slower, but repeatable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # small "a.b.c=value" override syntax for the command line
    for item in overrides or []:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_csv_row(path, row: dict):
    """Append one row to a CSV (writes the header the first time)."""
    import csv
    ensure_dir(Path(path).parent)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def environment_info():
    """Software / hardware versions, saved with the results for reproducibility."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return info


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_plot_style():
    """One consistent look for every figure in the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,   # keeps titles / legends from overlapping
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "legend.frameon": False,
        "lines.linewidth": 2,
    })
    return plt


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0
