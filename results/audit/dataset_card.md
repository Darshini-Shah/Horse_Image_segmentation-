# Dataset card — Weizmann Horse

* Source: Borenstein & Ullman, "Class-specific, top-down segmentation", ECCV 2002.
  Kaggle mirror: https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database
* Unit of observation: one side-view RGB photograph of a horse + one binary mask (1 = horse).
* Pairs found: 327 (unpaired files: 0, empty masks: 0)
* Original size: width 211-1280 px,
  height 148-960 px (all resized to 256x256).
* Horse pixel share (256x256): mean 0.252,
  range 0.052-0.432.
* Possible duplicate groups (average hash): 7
* Split (stratified by horse-size tercile, seed fixed): {'train': 228, 'val': 49, 'test': 50}
* Mean horse share per split: {'train': 0.25, 'val': 0.249, 'test': 0.26}
* Licence: research use; cite the original paper.
