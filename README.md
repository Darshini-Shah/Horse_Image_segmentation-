# Horse Segmentation with Skip Connections — Weizmann Horse

**Problem statement 3.** Segment the horse in side-view photographs. Compare an
encoder-decoder with **no skip connections** against the **same network with
the skip connections restored**, using one loss for both, and show test
photographs where the outline improves.

**Hypothesis.** Max-pooling in the encoder throws away exact positions, so a
decoder that only sees the 16x16 bottleneck has to "guess" the outline
(legs, ears, tail). Skip connections hand the high-resolution encoder
features back to the decoder, so the outline — and therefore precision,
recall, F1 and IoU — should improve on unseen test images.

## Models (one class, `src/models.py`)

| Key | Model | Skip connection | Decoder upsampling | Role |
|---|---|---|---|---|
| `baseline_noskip` | Baseline | none | bilinear ×2 + 1×1 conv | required baseline |
| `p1_concat_interp` | Proposed 1 | concatenation | bilinear ×2 + 1×1 conv | main proposed model |
| `p2_add_interp` | Proposed 2 | addition | bilinear ×2 + 1×1 conv | ablation: *how* to merge the skip |
| `p3_concat_tconv` | Proposed 3 | concatenation | 2×2 transposed conv | ablation: *how* to upsample |

Same depth (4), widths (32–512), blocks, loss, optimiser, augmentation,
split and epochs for all four. P2 has exactly the same number of parameters
as the baseline, so its gain cannot be explained by extra capacity.

## Loss (same for all models)

```
L_BCE  = -(1/N) Σ_i [ g_i log p_i + (1 - g_i) log(1 - p_i) ]
L_Dice = 1 - (2 Σ_i p_i g_i + ε) / (Σ_i p_i + Σ_i g_i + ε)
L      = L_BCE + L_Dice
```

BCE gives stable per-pixel gradients; Dice scores the overlap of the whole
mask (not dominated by background) and matches the F1/IoU we report. The
choice is not hard-coded: `tune` compares BCE, Dice and BCE+Dice on the
**validation** split and uses the winner for every final run.

## Protocol

1. **Prepare** — pair images/masks, resize to 256×256 (bilinear for images,
   nearest for masks), cache. Fixed 70/15/15 split (≈229/49/49), stratified
   by horse-size tercile, saved in `results/split.json` and reused by every
   model and seed. Normalisation mean/std computed on the **train split only**.
2. **Tune** (validation only) — A: loss selection on Baseline + P1.
   B: learning-rate sweep {3e-3, 1e-3, 3e-4} for all four; one shared LR is
   chosen (mean validation Dice) so every model has identical settings.
3. **Train** — 4 models × 3 seeds, AdamW + cosine LR, mixed precision,
   on-GPU augmentation (flip, scale, rotation, shift, colour jitter),
   best checkpoint by validation Dice, early stopping.
4. **Evaluate** — the test split is used once: precision, recall, F1, IoU
   per image (threshold 0.5), averaged over images, then mean ± std over seeds.
5. **Analyze** — tables, significance tests, all figures, `summary.md`.

## How to run

**Google Colab (recommended):** open `notebooks/Horse_Segmentation_Colab.ipynb`,
choose a T4 GPU and run the cells top to bottom. It uploads this folder,
downloads the dataset and runs everything.

**Locally / any GPU machine:**
```bash
pip install -r requirements.txt
# put the dataset under data/raw  (see data/README.md)
python main.py --stage all                               # full run, ~2-3 h on a T4
python main.py --stage all --config configs/quick.yaml   # ~20-30 min check
python main.py --stage analyze                           # redraw plots only
```
Finished runs are cached by name, so re-running a command resumes after a
disconnect. If you change training settings, use a new `--results_dir`
(otherwise old runs with the same name are reused).

Pipeline check without the real data:
```bash
python tools/make_dummy_dataset.py --out data/dummy --n 60
python main.py --stage all --config configs/quick.yaml --data_root data/dummy \
    --set data.cache=data/cache_dummy.npz --results_dir results_dummy
```

## Outputs (`results/`)

| File | What it shows |
|---|---|
| `tables/main_results.csv`, `summary.md` | P / R / F1 / IoU (+ boundary metrics), mean ± std over seeds |
| `figures/results_main_metrics.png` | grouped bar chart of the four metrics for the four models |
| `figures/outline_two_examples.png` | **the two test photographs where the outline is better** (chosen automatically) |
| `figures/training_curves.png` | train loss, val loss, val Dice per epoch |
| `figures/tuning_loss_selection.png`, `tuning_lr_sweep.png` | hyperparameter tuning on validation |
| `figures/eda_*.png`, `audit/` | dataset audit and EDA |
| `figures/architecture.png` | architecture diagram of all four models |
| `figures/qualitative_random.png`, `qualitative_failures.png` | random test masks and the hardest cases |
| `experiment_log.csv` | every run: model, seed, loss, LR, best epoch, val Dice, runtime |
| `split.json`, `normalization.json`, `config_used.json`, `environment.json` | reproducibility |

## Beyond the required scope ★

Everything marked `[EXTRA]` in the code goes beyond what problem statement 3 asks for:

1. **Two extra ablations (P2, P3)** — separate *whether* skip connections
   help from *how* they are merged (concat vs add) and *how* the decoder upsamples.
2. **Boundary metrics: Boundary F1 and HD95** (`src/metrics.py`). Pixel IoU
   is dominated by the horse's interior; these measure the outline directly,
   which is exactly the claim of the problem statement.
3. **Error location split** (`figures/error_near_outline.png`) — wrong pixels
   within 3 px of the true outline vs. elsewhere. Shows *where* skip
   connections help.
4. **Multiple seeds + statistics** (`tables/significance.csv`) — 3 training
   seeds (mean ± std), paired Wilcoxon signed-rank test and bootstrap 95% CI
   on the per-image differences.
5. **Skip knock-out probe** (`figures/skip_knockout.png`) — zero one skip
   level of the trained P1 at test time to see which resolution it relies on.
6. **Validation-based loss selection and LR sweep** with saved logs.
7. **Error analysis by horse size** (`figures/results_by_horse_size.png`)
   and the hardest test images.
8. **Efficiency table** — parameters, model size, inference time, training time.
9. **Leakage controls** — stratified split saved once, train-only
   normalisation, disjointness assertion, test used once.
10. **Architecture diagram** generated from the code.

## Folder structure

```
configs/            default.yaml (full run), quick.yaml (fast check)
src/
  data.py           pairing, resize + cache, stratified split, normalisation, GPU augmentation
  models.py         configurable encoder-decoder (skip: none/concat/add, up: interp/tconv)
  losses.py         BCE, Dice, BCE+Dice
  metrics.py        P/R/F1/IoU, Boundary F1, HD95, near-outline error
  engine.py         training loop (AMP, best-val checkpoint, early stop, resume)
  tuning.py         loss selection + LR sweep on validation
  evaluate.py       one-time test evaluation, efficiency, skip knock-out
  analysis.py       tables, statistics, figures, summary.md
  eda.py            dataset audit + EDA figures
  diagram.py        architecture figure
main.py             entry point (--stage prepare|tune|train|evaluate|analyze|all)
notebooks/          Colab notebook
tools/              synthetic dataset for a pipeline check
data/               dataset goes here (see data/README.md)
results/            generated tables, figures, logs, checkpoints
```

## Limitations

* 327 images in total → 49 test images; differences of ~1 IoU point can be
  within noise, which is why seeds, CIs and a paired test are reported.
* Resizing to 256×256 changes the aspect ratio (required by the problem statement).
* The threshold is fixed at 0.5; no test-time augmentation or pretrained encoder,
  to keep the comparison about the skip connections only.
* The skip knock-out is a probe of a trained network, not a retrained ablation.
