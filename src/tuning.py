"""
Hyperparameter tuning — uses the VALIDATION split only.

Stage A, loss selection: train the baseline and P1 with each candidate loss
(BCE, Dice, BCE+Dice) at a fixed learning rate. The loss with the highest
mean validation Dice over the two models is used for ALL final runs.
(Using both the baseline and the proposed model avoids picking a loss that
only suits the proposed network.)

Stage B, learning-rate sweep: with the chosen loss, train every model at
each candidate LR. By default ("shared") we pick the single LR with the best
mean validation Dice over all models, so that every model is trained with
exactly the same settings and only the architecture differs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import train_run
from .utils import MODELS, load_json, save_json


def run_id_for(stage, model, loss, lr, seed):
    return f"{stage}_{model}_{loss}_lr{lr:g}_s{seed}"


def tune(cfg, train, val, norm, device):
    tc = cfg["tuning"]
    out = Path(cfg["results_dir"])
    seed = tc.get("seed", 0)
    rows = []

    # ---------------- Stage A: loss -----------------
    print("\n=== Tuning stage A: loss selection (validation split) ===")
    for loss in tc["loss_candidates"]:
        for m in tc["loss_models"]:
            rid = run_id_for("tune", m, loss, tc["loss_lr"], seed)
            s = train_run(run_id=rid, model_key=m, seed=seed, loss_name=loss, lr=tc["loss_lr"],
                          epochs=tc["epochs"], cfg=cfg, train=train, val=val, norm=norm,
                          device=device, out_dir=out, stage="tune_loss")
            rows.append({**s, "tune_stage": "loss"})
    df_loss = pd.DataFrame([r for r in rows if r["tune_stage"] == "loss"])
    loss_table = df_loss.groupby("loss")["best_val_dice"].mean().sort_values(ascending=False)
    best_loss = loss_table.index[0]
    print(loss_table.to_string())
    print(f"--> selected loss: {best_loss}")

    # ---------------- Stage B: learning rate -----------------
    print("\n=== Tuning stage B: learning-rate sweep (validation split) ===")
    for lr in tc["lr_candidates"]:
        for m in MODELS:
            # identical config to a stage-A run -> the cached run is reused (no retraining)
            rid = run_id_for("tune", m, best_loss, lr, seed)
            s = train_run(run_id=rid, model_key=m, seed=seed, loss_name=best_loss, lr=lr,
                          epochs=tc["epochs"], cfg=cfg, train=train, val=val, norm=norm,
                          device=device, out_dir=out, stage="tune_lr")
            rows.append({**s, "tune_stage": "lr"})
    df_lr = pd.DataFrame([r for r in rows if r["tune_stage"] == "lr"])
    pivot = df_lr.pivot_table(index="lr", columns="model", values="best_val_dice")
    print(pivot.to_string())

    if tc.get("lr_selection", "shared") == "shared":
        best_lr_shared = float(pivot.mean(axis=1).idxmax())
        chosen_lr = {m: best_lr_shared for m in MODELS}
    else:  # per model, same search budget for every model
        chosen_lr = {m: float(pivot[m].idxmax()) for m in MODELS}
    print(f"--> selected learning rate(s): {chosen_lr}")

    pd.DataFrame(rows).to_csv(out / "tables" / "tuning_runs.csv", index=False)
    result = {"loss": best_loss, "lr": chosen_lr,
              "loss_table": loss_table.to_dict(),
              "lr_table": {str(k): v for k, v in pivot.to_dict(orient="index").items()},
              "lr_selection": tc.get("lr_selection", "shared")}
    save_json(result, out / "tuning_result.json")
    return result


def load_or_default_tuning(cfg):
    p = Path(cfg["results_dir"]) / "tuning_result.json"
    if p.exists():
        return load_json(p)
    # tuning disabled -> fall back to config values
    return {"loss": cfg["train"]["loss"], "lr": {m: cfg["train"]["lr"] for m in MODELS},
            "lr_selection": "config"}
