"""
Training and inference loops.

One "run" = (model, loss, learning rate, seed). For every run we
  * train on the TRAIN split only,
  * evaluate Dice/IoU on the VAL split after every epoch,
  * keep the checkpoint with the best validation Dice (model selection
    never looks at the test split),
  * log the per-epoch history to CSV and one summary row to the experiment log.
Runs that already have a finished checkpoint are skipped, so if Colab
disconnects you can just re-run the same command and it resumes.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import GPUAugment
from .losses import SegLoss
from .metrics import batch_dice_iou
from .models import build_model
from .utils import (Timer, append_csv_row, count_params, ensure_dir, load_json,
                    save_json, set_seed)


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict_probs(model, seg, norm, device, amp=True, bs=16, drop_skips=()):
    """Sigmoid probabilities for every image of a split -> (N, H, W) float32 numpy."""
    model.eval()
    out = []
    for i in range(0, len(seg), bs):
        x = norm(seg.images[i:i + bs].float() / 255.0)
        with _autocast(device, amp):
            logits = model(x, drop_skips=drop_skips)
        out.append(torch.sigmoid(logits.float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def evaluate_split(model, seg, norm, loss_fn, device, amp=True, bs=16, thr=0.5):
    model.eval()
    losses, dices, ious = [], [], []
    for i in range(0, len(seg), bs):
        x = norm(seg.images[i:i + bs].float() / 255.0)
        y = seg.masks[i:i + bs].float()
        with _autocast(device, amp):
            logits = model(x)
        losses.append(loss_fn(logits, y).item() * x.shape[0])
        d, j = batch_dice_iou(torch.sigmoid(logits.float()) >= thr, y)
        dices.append(d.cpu())
        ious.append(j.cpu())
    n = len(seg)
    return sum(losses) / n, torch.cat(dices).mean().item(), torch.cat(ious).mean().item()


def train_run(*, run_id, model_key, seed, loss_name, lr, epochs, cfg, train, val, norm,
              device, out_dir, stage):
    """Train one configuration. Returns a summary dict (also written to the experiment log)."""
    out_dir = ensure_dir(out_dir)
    ckpt = out_dir / "checkpoints" / f"{run_id}.pt"
    summ_path = out_dir / "runs" / f"{run_id}.json"
    if ckpt.exists() and summ_path.exists():
        print(f"  [skip] {run_id} already finished")
        return load_json(summ_path)

    tcfg = cfg["train"]
    set_seed(seed)
    model = build_model(model_key, cfg).to(device)
    loss_fn = SegLoss(loss_name, tcfg.get("w_bce", 1.0), tcfg.get("w_dice", 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    # cosine decay of the learning rate over the planned number of epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    amp = tcfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    g = torch.Generator().manual_seed(seed)
    augment = GPUAugment(tcfg["augment"], g)
    bs = tcfg["batch_size"]
    n = len(train)
    patience = tcfg.get("patience", epochs)

    best = {"val_dice": -1.0, "epoch": -1}
    best_state, history, bad_epochs = None, [], 0
    with Timer() as timer:
        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n, generator=g)
            run_loss = 0.0
            # drop a final batch of size 1 (BatchNorm cannot use it)
            n_used = n - (n % bs) if n % bs == 1 else n
            for i in range(0, n_used, bs):
                idx = perm[i:i + bs].to(device)
                x = train.images[idx].float() / 255.0
                y = train.masks[idx].float()
                x, y = augment(x, y)
                x = norm(x)
                with _autocast(device, amp):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item() * x.shape[0]
            sched.step()

            v_loss, v_dice, v_iou = evaluate_split(model, val, norm, loss_fn, device, amp)
            history.append(dict(epoch=epoch, train_loss=run_loss / n_used, val_loss=v_loss,
                                val_dice=v_dice, val_iou=v_iou, lr=opt.param_groups[0]["lr"]))
            if v_dice > best["val_dice"]:
                best = {"val_dice": v_dice, "val_iou": v_iou, "epoch": epoch}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"  {run_id} | ep {epoch:3d} | train {run_loss / n_used:.4f} | "
                      f"val loss {v_loss:.4f} dice {v_dice:.4f} iou {v_iou:.4f}")
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (best epoch {best['epoch']})")
                break

    ensure_dir(ckpt.parent)
    torch.save({"state_dict": best_state, "model_key": model_key, "seed": seed,
                "loss": loss_name, "lr": lr, "cfg_model": cfg["model"]}, ckpt)
    hist_path = out_dir / "histories" / f"{run_id}.csv"
    ensure_dir(hist_path.parent)
    pd.DataFrame(history).to_csv(hist_path, index=False)

    summary = dict(run_id=run_id, stage=stage, model=model_key, seed=seed, split_seed=cfg["split_seed"],
                   loss=loss_name, lr=lr, batch_size=bs, epochs_planned=epochs,
                   epochs_run=len(history), best_epoch=best["epoch"],
                   best_val_dice=round(best["val_dice"], 5), best_val_iou=round(best["val_iou"], 5),
                   params=count_params(model), runtime_s=round(timer.elapsed, 1))
    save_json(summary, summ_path)
    # experiment log in the format suggested in the project guidelines
    append_csv_row(Path(cfg["results_dir"]) / "experiment_log.csv", summary)
    return summary


def load_trained(run_id, cfg, device):
    ck = torch.load(Path(cfg["results_dir"]) / "checkpoints" / f"{run_id}.pt", map_location=device)
    model = build_model(ck["model_key"], cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
