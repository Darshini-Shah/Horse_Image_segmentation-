"""
Horse segmentation (Weizmann Horse) — effect of skip connections in an
encoder-decoder. Single entry point for the whole pipeline.

Stages (run in this order, or all at once with --stage all):
  prepare   pair + resize images, fixed stratified split, dataset audit / EDA,
            architecture diagram
  tune      loss selection + learning-rate sweep on the VALIDATION split
  train     final training of the 4 models x N seeds with the chosen settings
  evaluate  one-time evaluation on the TEST split (+ skip knock-out probe)
  analyze   tables, significance tests, figures, summary.md

Examples
  python main.py --stage all
  python main.py --stage all --config configs/quick.yaml        # ~15 min sanity run
  python main.py --stage train --set train.epochs=80 seeds=[0,1]
  python main.py --stage all --results_dir /content/drive/MyDrive/horse_results
"""
import argparse
from pathlib import Path

from src.data import (Normalizer, SegData, build_cache, make_or_load_split,
                      train_norm_stats)
from src.utils import (MODELS, ensure_dir, environment_info, get_device,
                       load_config, save_json)

STAGES = ["prepare", "tune", "train", "evaluate", "analyze"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--data_root", default=None, help="folder that contains the dataset")
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. train.epochs=50")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    out = ensure_dir(cfg["results_dir"])
    for sub in ("tables", "figures", "checkpoints", "histories", "runs", "predictions"):
        ensure_dir(out / sub)
    save_json(cfg, out / "config_used.json")
    save_json(environment_info(), out / "environment.json")
    device = get_device()
    print(f"Device: {device} | results -> {out}")

    stages = STAGES if args.stage == "all" else [args.stage]

    # ---- data: always needed (cheap after the first time thanks to the cache) ----
    data = build_cache(cfg)
    split = make_or_load_split(data, cfg, out / "split.json")
    mean, std = train_norm_stats(data, split)
    save_json({"mean": mean, "std": std, "note": "computed on the training split only"},
              out / "normalization.json")
    norm = Normalizer(mean, std, device)
    train = SegData(data, split["train"], device)
    val = SegData(data, split["val"], device)
    test = SegData(data, split["test"], device)
    print(f"Split sizes: train {len(train)} | val {len(val)} | test {len(test)}")

    if "prepare" in stages:
        from src.diagram import draw_architectures
        from src.eda import run_audit
        run_audit(data, split, cfg)
        draw_architectures(out / "figures" / "architecture.png")

    from src.tuning import load_or_default_tuning, tune
    if "tune" in stages and cfg["tuning"]["enabled"]:
        tuning = tune(cfg, train, val, norm, device)
    else:
        tuning = load_or_default_tuning(cfg)

    if "train" in stages:
        from src.engine import train_run
        from src.evaluate import final_run_id
        print(f"\n=== Final training: loss={tuning['loss']} lr={tuning['lr']} seeds={cfg['seeds']} ===")
        for seed in cfg["seeds"]:
            for m in MODELS:
                lr = tuning["lr"][m]
                train_run(run_id=final_run_id(m, tuning["loss"], lr, seed), model_key=m, seed=seed,
                          loss_name=tuning["loss"], lr=lr, epochs=cfg["train"]["epochs"], cfg=cfg,
                          train=train, val=val, norm=norm, device=device, out_dir=out, stage="final")

    if "evaluate" in stages:
        from src.evaluate import evaluate_all
        print("\n=== Test-set evaluation ===")
        evaluate_all(cfg, data, split, tuning, test, norm, device)

    if "analyze" in stages:
        from src.analysis import analyze
        print("\n=== Analysis ===")
        analyze(cfg, data, split, tuning)


if __name__ == "__main__":
    main()
