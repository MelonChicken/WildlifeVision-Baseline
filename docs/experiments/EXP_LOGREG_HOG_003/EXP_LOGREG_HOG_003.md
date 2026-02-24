### Experiment ID

* `EXP_LOGREG_HOG_003` (tag: `logreg_hog_grid_v4_focus_bal_ns6`, best run_id: `5e699b8381`)

### Setting

* Feature: HOG
  * `pixels_per_cell=(8, 8)`
  * `cells_per_block=(2, 2)`
  * `orientations=9`
  * `block_norm="L2-Hys"`
* Model: LogisticRegression
  * `C=0.003`
  * `use_scaler=False`
  * `class_weight="balanced"`
  * `max_iter=6000`
  * `solver="lbfgs"`

### Validation

* `GroupKFold(n_splits=6), group=site`

### Result

* Fold log loss:
  * `[1.937881, 1.962835, 1.913209, 1.924022, 1.696595, 1.699670]`
* Mean log loss:
  * `1.855702` (std `0.112447`)
