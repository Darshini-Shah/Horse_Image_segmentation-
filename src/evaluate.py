"""
Test-set evaluation. This is the ONLY place where the test split is used,
and it is used once, after model selection on the validation split.

Outputs (results/tables/):
  per_image_metrics.csv   one row per (model, seed, test image)
  efficiency.csv          parameters, model size, inference time, training time
  skip_knockout.csv       [EXTRA] test IoU when individual skip connections are zeroed
and the raw test probabilities in results/predictions/ (used by the plots).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .engine import load_trained, predict_probs
from .metrics import all_metrics, confusion, region_metrics
from .utils import MODELS, count_params, ensure_dir, load_json


def final_run_id(model_key, loss, lr, seed):
    return f"final_{model_key}_{loss}_lr{lr:g}_s{seed}"


@torch.no_grad()
def inference_ms(model, device, size=256, reps=50):
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def evaluate_all(cfg, data, split, tuning, test, norm, device):
    out = Path(cfg["results_dir"])
    tab, pred_dir = ensure_dir(out / "tables"), ensure_dir(out / "predictions")
    ev = cfg["eval"]
    thr, tol, band = ev["threshold"], ev["boundary_tol"], ev["band_width"]
    gts = data["masks"][split["test"]]
    fg = gts.reshape(len(gts), -1).mean(1)
    names = [str(data["names"][i]) for i in split["test"]]
    loss = tuning["loss"]

    rows, eff, knock = [], [], []
    for m in MODELS:
        lr = tuning["lr"][m]
        for seed in cfg["seeds"]:
            rid = final_run_id(m, loss, lr, seed)
            model = load_trained(rid, cfg, device)
            probs = predict_probs(model, test, norm, device, cfg["train"].get("amp", True))
            np.save(pred_dir / f"{m}_s{seed}.npy", (probs * 255).round().astype(np.uint8))

            for j in range(len(gts)):
                met = all_metrics(probs[j], gts[j], thr, tol, band)
                tp, fp, fn, _ = confusion(probs[j] >= thr, gts[j])
                rows.append(dict(model=m, seed=seed, test_pos=j, name=names[j], fg_share=fg[j],
                                 tp=int(tp), fp=int(fp), fn=int(fn), **met))

            run = load_json(out / "runs" / f"{rid}.json")
            eff.append(dict(model=m, seed=seed, params=count_params(model),
                            size_mb=count_params(model) * 4 / 2 ** 20,
                            infer_ms=inference_ms(model, device),
                            train_time_s=run["runtime_s"], best_epoch=run["best_epoch"],
                            best_val_dice=run["best_val_dice"]))

            # [EXTRA] Skip knock-out: zero one skip level at test time and see how much
            # the trained model relies on it. Level 1 = full resolution (fine outline).
            # Note: this is a probe of a trained network (the model never saw zeroed
            # skips), not a retrained ablation — the retrained ablation is the baseline.
            if MODELS[m][1] != "none":
                depth = cfg["model"]["depth"]
                variants = [("none dropped", ())] + [(f"drop L{l}", (l,)) for l in range(1, depth + 1)] \
                           + [("drop all", tuple(range(1, depth + 1)))]
                for label, drop in variants:
                    pk = predict_probs(model, test, norm, device, cfg["train"].get("amp", True),
                                       drop_skips=drop)
                    ious = [region_metrics(pk[j] >= thr, gts[j])["iou"] for j in range(len(gts))]
                    knock.append(dict(model=m, seed=seed, variant=label, iou=float(np.mean(ious))))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  evaluated {rid}")

    pd.DataFrame(rows).to_csv(tab / "per_image_metrics.csv", index=False)
    pd.DataFrame(eff).to_csv(tab / "efficiency.csv", index=False)
    pd.DataFrame(knock).to_csv(tab / "skip_knockout.csv", index=False)
    return pd.DataFrame(rows)
