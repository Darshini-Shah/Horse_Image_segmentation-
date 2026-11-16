# Complete In-Depth Codebase & Architecture Guide: Horse Segmentation

This document provides a comprehensive, file-by-file breakdown of the entire codebase for **Problem Statement 3: Horse Segmentation (Weizmann Horse Dataset)**. It explains what each component does, why it is designed that way, how data flows through the system, and how to interpret the results for reports or viva evaluations.

---

## 1. High-Level Blueprint & Architecture

The primary goal of this project is to investigate whether adding **skip connections** to an encoder-decoder network improves semantic segmentation performance on the Weizmann Horse Database (327 images), specifically on **fine boundary delineation (legs, hooves, tail, muzzle, ears)**.

```mermaid
flowchart TD
    A["Raw Images & Masks\n(data/horse, data/mask)"] --> B["src/data.py\n(Pair, Resize 256x256, Cache .npz)"]
    B --> C["src/eda.py\n(Dataset Audit & EDA Figures)"]
    B --> D["src/tuning.py\n(Val-Split Loss & LR Grid Search)"]
    D --> E["src/engine.py\n(Train 4 Models on Train Split)"]
    E --> F["src/evaluate.py\n(Single-pass Test Evaluation + Knockout)"]
    F --> G["src/analysis.py\n(Wilcoxon Stats, Tables, 10 Figures, summary.md)"]
    H["main.py\n(CLI Orchestrator)"] -. controls .-> B & C & D & E & F & G
```

### The 4 Models Compared (`src/models.py`)

All four models share the **exact same** 4-stage encoder depth (channel widths: $32 \to 64 \to 128 \to 256$), the same convolution blocks, and identical parameter capacities (~1.8M to 2.1M parameters). The **only controlled variable** is the presence and type of skip connections:

| Model ID | Skip Connection | Upsampling Type | Description |
|---|:---:|:---:|---|
| **Baseline (`baseline_noskip`)** | **None** | Bilinear + $1\times 1$ Conv | Standard hourglass encoder-decoder. High-frequency spatial detail must squeeze through the bottleneck. |
| **Proposed 1 (`p1_concat_interp`)** | **Concatenation** | Bilinear + $1\times 1$ Conv | Classic **U-Net style**: encoder features are concatenated along the channel dimension. |
| **Proposed 2 (`p2_add_interp`)** | **Addition** | Bilinear + $1\times 1$ Conv | **ResNet / FPN style**: encoder features are added element-wise to upsampled features (zero parameter overhead). |
| **Proposed 3 (`p3_concat_tconv`)** | **Concatenation** | Transposed Conv ($2\times 2$) | U-Net skip connections with learnable deconvolution kernels instead of fixed bilinear interpolation. |

---

## 2. Directory Structure

```
MLproject_rollno_horse_segmentation/
├── main.py                     # Master CLI entry point & pipeline coordinator
├── code_explain.md             # In-depth architectural & codebase documentation
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & assignment specifications
├── configs/
│   ├── default.yaml            # Full experiment config (3 seeds, 120 epochs)
│   └── quick.yaml              # Quick config (1 seed, 40 epochs, ~25 min)
├── src/
│   ├── models.py               # PyTorch neural network architectures
│   ├── data.py                 # Loading, caching, stratification & augmentation
│   ├── losses.py               # BCE, Dice, and compound BCE+Dice losses
│   ├── metrics.py              # IoU, F1, Precision, Recall, Boundary F1, HD95
│   ├── engine.py               # Training loops, validation checks, early stopping
│   ├── tuning.py               # Hyperparameter search (Validation split only)
│   ├── evaluate.py             # Test split evaluation & skip knockout ablation
│   ├── analysis.py             # Hypothesis testing & all 10 report plots
│   ├── eda.py                  # Dataset exploratory analysis & summary cards
│   ├── diagram.py              # Generates the architecture schematic figure
│   └── utils.py                # Seeds, device selection, config parsers, plot styling
├── notebooks/
│   └── Horse_Segmentation_Colab.ipynb # Self-contained Google Colab notebook
├── tools/
│   ├── build_colab_notebook.py # Re-packs source code into base64 notebook payload
│   └── make_dummy_dataset.py   # Generates synthetic data for local testing
└── results_quick/              # Outputs produced by the quick pipeline run
    ├── summary.md              # Markdown report summarizing all findings
    ├── figures/                # 10 publication-quality PNG figures
    ├── tables/                 # 5 CSV metric and significance tables
    └── histories/              # Per-epoch training & validation logs
```

---

## 3. In-Depth File-by-File Breakdown

### `main.py` — Master Orchestrator
- **What it does:** Serves as the central command-line interface (CLI) coordinator that executes each phase of the project in proper sequence.
- **Key Functions:**
  - `parse_args()`: Parses user flags such as `--stage` (`prepare`, `tune`, `train`, `evaluate`, `analyze`, or `all`), `--config`, and `--results_dir`.
  - Sequential dispatching:
    1. `run_prepare()`: Invokes data discovery, caching, stratified splitting, EDA, and architecture diagram generation.
    2. `run_tuning()`: Executes grid search on the validation set for loss function and learning rate.
    3. `run_train()`: Trains the 4 architectures across specified seeds using the optimal hyperparameters.
    4. `run_evaluate()`: Runs single-pass evaluation on the untouched test split, along with the skip-connection knockout ablation.
    5. `run_analyze()`: Computes statistical significance tests and renders all 10 figures and the executive summary.

---

### `src/models.py` — PyTorch Neural Network Architectures
- **What it does:** Implements the complete family of encoder-decoder models in a unified, modular class.
- **Key Classes & Methods:**
  - `ConvBlock(in_c, out_c)`: A standard double-convolution block: $\text{Conv3x3} \to \text{BatchNorm2d} \to \text{ReLU} \to \text{Conv3x3} \to \text{BatchNorm2d} \to \text{ReLU}$.
  - `Encoder(in_c, base_c, depth)`: Computes progressive downsampling via $2\times 2$ `MaxPool2d` while storing intermediate feature maps at each spatial resolution ($256 \to 128 \to 64 \to 32 \to 16$).
  - `DecoderStage(in_c, skip_c, out_c, skip_mode, up_mode)`: Handles upsampling and feature fusion:
    - If `up_mode == "interp"`: Bilinear $2\times$ interpolation followed by a $1\times 1$ conv.
    - If `up_mode == "tconv"`: Transposed $2\times 2$ convolution (`nn.ConvTranspose2d(..., stride=2)`).
    - If `skip_mode == "none"`: Standard feedforward (upsampled tensor only).
    - If `skip_mode == "concat"`: Channel concatenation (`torch.cat([upsampled, skip_feat], dim=1)`).
    - If `skip_mode == "add"`: Element-wise addition (`upsampled + skip_feat`).
  - `EncoderDecoder`: The unified model wrapper. Crucially includes a `knockout_skips=[stage_idx]` parameter that zeros out specific skip connections during inference to measure their individual impact.
  - `build_model(name, ...)`: Factory helper to instantiate the four specific models (`baseline_noskip`, `p1_concat_interp`, `p2_add_interp`, `p3_concat_tconv`).

---

### `src/data.py` — Data Loading, Caching & Augmentation
- **What it does:** Discovers raw horse images and masks, pairs them, resizes them, caches them into memory, and generates balanced splits with on-GPU augmentation.
- **Key Functions:**
  - `discover_pairs(root)`: Scans the dataset directory, matching images (`horse*.png`) with their corresponding masks.
  - `build_cache(pairs, cache_path, img_size=256)`: Pre-resizes all 327 image-mask pairs to $256 \times 256$ and saves them into a compressed `.npz` file (~64 MB total). This allows the entire dataset to reside in RAM/VRAM, eliminating disk I/O bottlenecks during training.
  - `make_or_load_split(...)`: Performs a **stratified 70% / 15% / 15% train/val/test split** (228 train, 49 val, 50 test). The stratification bins images by **foreground horse pixel ratio**, ensuring small, medium, and large horses are equally represented across all sets.
  - `compute_train_norm(...)`: Computes dataset mean and standard deviation **strictly on the training split** to prevent data leakage.
  - `augment_batch(...)`: Fast, vectorized GPU augmentation applied on-the-fly during training:
    - Random horizontal flip ($p=0.5$).
    - Random affine scaling ($0.85\times - 1.15\times$).
    - Random rotation ($\pm 10^\circ$).
    - Translation and color jitter (brightness, contrast, saturation).

---

### `src/losses.py` — Loss Functions
- **What it does:** Provides the loss objectives evaluated during hyperparameter tuning and model training.
- **Formulas & Implementations:**
  - `BCELossWrapper`: Standard Binary Cross-Entropy:
    $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \right]$$
  - `SoftDiceLoss`: Differentiable soft Dice loss operating on continuous sigmoid predictions:
    $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}$$
  - `CompoundBCEDiceLoss`: The winning combined objective:
    $$\mathcal{L} = \lambda_{\text{bce}}\mathcal{L}_{\text{BCE}} + \lambda_{\text{dice}}\mathcal{L}_{\text{Dice}} \quad (\lambda_{\text{bce}}=1.0, \lambda_{\text{dice}}=1.0)$$
    *BCE provides stable gradient flow across background pixels, while Dice directly counteracts foreground-background class imbalance.*

---

### `src/metrics.py` — Comprehensive & Boundary Evaluation Metrics
- **What it does:** Computes image-level segmentation and boundary metrics on binarized predictions (threshold = 0.5).
- **Core Region Metrics:**
  - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
  - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
  - $\text{F1 / Dice} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$
  - $\text{IoU (Jaccard Index)} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$
- **Boundary Metrics (Core to the Skip Connection Hypothesis):**
  - `boundary_f1()`: Extracts 1-pixel boundary contours using morphological dilation/erosion and computes F1 within a 2-pixel tolerance band.
  - `hausdorff95()`: 95th-percentile Hausdorff Distance measuring maximum boundary localization errors (e.g., thin leg tips, ears, tail).
  - `near_outline_error()`: Calculates what percentage of total prediction errors occur within a 3-pixel band of the ground-truth boundary.

---

### `src/tuning.py` — Validation-Only Hyperparameter Tuning
- **What it does:** Conducts hyperparameter optimization strictly using the validation split. The test split is never touched during this stage.
- **Workflow:**
  - **Stage A (Loss Selection):** Trains Baseline and P1 for 10–20 epochs across candidate losses (`bce`, `dice`, `bce_dice`). Selects the loss achieving highest mean validation Dice across both models (`bce_dice`).
  - **Stage B (Learning Rate Sweep):** Evaluates candidate learning rates (`0.001`, `0.0003`) on the validation set to find the optimal shared learning rate (`0.0003`).

---

### `src/engine.py` — Training Loop & Validation Checkpointing
- **What it does:** Coordinates training and validation loops with reproducibility and efficiency features.
- **Key Capabilities:**
  - **Mixed Precision:** Uses PyTorch `torch.cuda.amp.autocast()` and `GradScaler` for memory efficiency and accelerated training on GPUs.
  - **Validation Monitoring:** Evaluates validation Dice at the conclusion of every epoch.
  - **Best-Checkpoint Preservation:** Automatically tracks and saves the checkpoint corresponding to the highest validation score (`best.pt`).
  - **Resume Capability:** If a run's checkpoint already exists on disk, it is automatically skipped to prevent redundant compute.

---

### `src/evaluate.py` — Test Split Evaluation & Ablation
- **What it does:** Evaluates trained checkpoints on the untouched 50-image test set.
- **Outputs Produced:**
  - `per_image_metrics.csv`: Metric records for every single test image.
  - `efficiency.csv`: Model parameters, disk size, inference latency (ms), and training duration.
  - `skip_knockout.csv`: Evaluates P1 (U-Net) while selectively disabling individual skip connections (`knockout_skips=[i]`) to quantify each level's contribution to boundary sharpness.

---

### `src/analysis.py` — Statistics & Visualization Generation
- **What it does:** Reads evaluation CSVs and generates all statistical significance tests, markdown summaries, and 10 publication-quality figures.
- **Statistical Tests:**
  - Paired Student’s t-test and **Wilcoxon signed-rank test** comparing Baseline vs. P1, P2, and P3 across all test images.
  - Bootstrap 95% Confidence Intervals (2,000 resamples).
- **Generated Figures (`results_quick/figures/`):**
  1. `architecture.png`: Visual layout of all 4 network designs.
  2. `training_curves.png`: Training loss and validation Dice trajectories across epochs.
  3. `results_main_metrics.png`: Bar plots with error bars comparing IoU, F1, and Boundary F1.
  4. `outline_two_examples.png`: Qualitative boundary overlays highlighting where Baseline fails and P1 succeeds.
  5. `error_near_outline.png`: Spatial breakdown of errors inside, outside, and along the outline.
  6. `skip_knockout.png`: Ablation chart showing performance drops when individual skip levels are zeroed out.
  7. `results_by_horse_size.png`: Performance breakdown by horse size (small vs medium vs large).
  8. `qualitative_random.png`: 6 random test predictions comparing all models against Ground Truth.
  9. `qualitative_failures.png`: Hardest test cases (complex backgrounds, occluded legs).
  10. `efficiency.png`: Accuracy vs parameter count and latency trade-off.

---

### `src/eda.py` — Exploratory Data Analysis
- **What it does:** Performs an automated dataset audit prior to training.
- **Produces:**
  - `audit.json`: Total count, duplicate detection, image dimensions, and horse-to-background pixel ratios.
  - `dataset_card.md`: Formatted dataset card for inclusion in academic reports.
  - EDA figures: Size distributions, aspect ratios, and spatial heatmaps ("mean horse mask").

---

### `src/diagram.py` — Programmatic Architecture Drawing
- **What it does:** Renders `architecture.png` programmatically using `matplotlib.patches`, visually illustrating the exact tensor dimensions, channel widths, downsampling, and skip routing across all 4 architectures.

---

### `src/utils.py` — Shared Utilities & Style Definitions
- **What it does:** Central repository for shared utility functions:
  - `set_seed(seed)`: Sets deterministic seeds for Python `random`, NumPy, and PyTorch (`torch.backends.cudnn.deterministic = True`).
  - `get_device()`: Automatically selects CUDA GPU, Apple Silicon MPS, or CPU.
  - `setup_plot_style()`: Configures high-DPI matplotlib styling (300 DPI, clean fonts, consistent color schemes).

---

### Configuration Files (`configs/`)
- **`configs/quick.yaml`**: The fast configuration (~25 min on Colab GPU). Uses 1 seed (`seed: 0`), 40 epochs, and a compact grid search. Ideal for verification, coursework submission, and generating all tables/plots.
- **`configs/default.yaml`**: The full configuration (3 seeds, 120 epochs, ~4–5 hours on GPU). Designed for multi-seed variance reporting.

---

### `notebooks/Horse_Segmentation_Colab.ipynb` — Google Colab Notebook
- **What it does:** Provides a self-contained, turnkey execution environment designed for Google Colab or VS Code's Colab extension.
- **Workflow in Notebook:**
  1. Checks GPU status (`!nvidia-smi`).
  2. Auto-restores project source code from an embedded base64 payload into the Colab workspace.
  3. Auto-downloads the Weizmann Horse dataset via `kagglehub`.
  4. Executes the full pipeline (`main.py --config configs/quick.yaml`).
  5. Displays `summary.md` and renders all 10 output figures inline.
  6. Bundles tables and figures into `horse_results.zip` for one-click downloading.

---

### Tools Directory (`tools/`)
- **`tools/make_dummy_dataset.py`**: Generates synthetic geometric horses for local unit-testing without requiring the actual dataset.
- **`tools/build_colab_notebook.py`**: Zips `src/`, `configs/`, and `main.py`, converts them into a base64 payload, and generates the self-contained `.ipynb` notebook.

---

## 4. End-to-End Data Flow Lifecycle

```
1. PREPARE:
   data/horse/ + data/mask/ ──[src/data.py]──> data/cache_256.npz (256x256 uint8)
                                       └──> results/split.json (70/15/15 stratified)

2. TUNE:
   Train split (228 images) ──[src/tuning.py]──> Evaluated on Val split (49 images)
                                       └──> Best loss = bce_dice, Best LR = 0.0003

3. TRAIN:
   Train split ──[src/engine.py]──> Train 4 models (40 epochs, AdamW + AMP)
                             └──> Checkpoints saved based on peak Val Dice

4. EVALUATE:
   Test split (50 images) ──[src/evaluate.py]──> results/tables/main_results.csv
                                        └──> results/tables/skip_knockout.csv

5. ANALYZE:
   CSVs ──[src/analysis.py]──> 10 PNG figures in results/figures/
                       └──> results/summary.md + significance.csv
```

---

## 5. Key Questions & Findings for Viva / Evaluation

### Q1: Why is the IoU of Baseline (No Skip) so close to U-Net (87.81% vs 89.97%)?
- **Answer:** The horse body interior constitutes ~95% of all horse pixels. Both architectures easily classify the large interior region. Pixel-level IoU is mathematically dominated by large contiguous areas, masking subtle errors along fine boundaries.

### Q2: How do you prove skip connections actually helped?
- **Answer:** Look at the **boundary-specific metrics**:
  - **Boundary F1 Score:** Jumps from **76.18%** (Baseline) to **83.51%** (P1 Concat) — a **+7.33% improvement**.
  - **Near-Outline Error:** Decreases from **2.50%** to **1.89%** — a **24.4% relative error reduction** along the border.
  - **Wilcoxon Signed-Rank Test:** Yields $p = 3.4 \times 10^{-14}$, confirming with extreme statistical certainty that the improvement is not random noise.

### Q3: What is the difference between Quick run and Full run?
- **Quick Run (`configs/quick.yaml`):** 1 seed, 40 epochs, 4 tuning runs ($\approx 200$ total epochs, ~25 mins on GPU).
- **Full Run (`configs/default.yaml`):** 3 seeds, 120 epochs, 9 tuning runs ($\approx 1,620$ total epochs, ~4–5 hours on GPU).
- **Conclusion:** Both runs arrive at the exact same scientific conclusions. The only difference is that the full run provides multi-seed standard deviation error bars ($\pm \text{std}$).

