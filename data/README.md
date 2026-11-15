# Data

The dataset is **not** included in this folder (licence + size). Download the
Weizmann Horse database (327 side-view photos + binary masks):

* Kaggle: https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database
* Original paper: E. Borenstein and S. Ullman, "Class-specific, top-down segmentation", ECCV 2002.

Put it so that the images and masks are somewhere under `data/raw/`, e.g.

```
data/raw/weizmann_horse_db/horse/horse001.png ...
data/raw/weizmann_horse_db/mask/horse001.png  ...
```

The loader searches `data/raw/` recursively and pairs files by name, so the
exact nesting does not matter (the original `rgb/` + `figure_ground/` layout
also works; `gray/` is ignored).

On the first run `main.py` writes `data/cache_256.npz` (all images resized to
256x256). Delete it if you change the dataset or `img_size`.

In Colab the notebook downloads the data for you (see `notebooks/`).
