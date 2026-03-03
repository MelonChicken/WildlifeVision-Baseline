import numpy as np

from src.config.global_variables import CLASS_COLS
from src.config.paths import LOG_PATH, PROJECT_ROOT
from src.data.make_table import make_train_table
from src.features.hog_matrix import build_hog_features
from src.models.model_families import CFG_A, MODEL_FAMILIES, build_model_pipeline, get_model_params
from src.training.crossvalidation import (
    add_group_folds,
    check_no_site_overlap_between_train_valid,
    check_site_has_single_fold,
)
from src.training.evaluate import evaluate_by_fold
from src.training.experiment_logger import log_experiment
from src.training.experiment_schema import CVChecks, DataSignature, HogParams, LogRegParams, build_record

# 1) Build train table
_df_train_table = make_train_table(autosave=True)

# 2) Add GroupKFold(site)
df_train_table_w_folds = add_group_folds(
    _df_train_table,
    n_splits=5,
    shuffle=True,
    random_state=42,
    autosave=True,
)

# 3) Validate fold integrity
check_site_has_single_fold(df_train_table_w_folds, site_col="site", fold_col="fold")
check_no_site_overlap_between_train_valid(df_train_table_w_folds, site_col="site", fold_col="fold")
fold_site_counts = {
    int(k): int(v)
    for k, v in df_train_table_w_folds.groupby("fold")["site"].nunique().to_dict().items()
}

# 4) Build labels and fold vectors
row_sum = df_train_table_w_folds[CLASS_COLS].sum(axis=1)
assert (row_sum == 1).all(), "Each sample must have exactly one active class."
y_idx = df_train_table_w_folds[CLASS_COLS].to_numpy().argmax(axis=1)
y = np.array([CLASS_COLS[i] for i in y_idx], dtype=object)
fold = df_train_table_w_folds["fold"].to_numpy()

class_counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
data_sig = DataSignature(
    n_samples=int(len(y)),
    n_sites=int(df_train_table_w_folds["site"].nunique()),
    class_counts=class_counts,
)

# 5) Fixed HOG params (orientation best=16)
orientation = 16
hog_params = HogParams(
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=orientation,
    block_norm="L2-Hys",
)

X = build_hog_features(
    df_train_table_w_folds,
    project_dir=PROJECT_ROOT,
    pixels_per_cell=hog_params.pixels_per_cell,
    cells_per_block=hog_params.cells_per_block,
    orientations=hog_params.orientations,
    block_norm=hog_params.block_norm,
)

# 6) Sweep model families
for model_family in MODEL_FAMILIES:
    mean_log_loss, standard_log_loss, scores = evaluate_by_fold(
        X=X,
        y=y,
        fold=fold,
        build_model_fn=lambda mf=model_family: build_model_pipeline(mf),
    )

    model_params = get_model_params(model_family)
    logreg_params = (
        LogRegParams(
            C=float(CFG_A["C"]),
            max_iter=int(CFG_A["max_iter"]),
            use_scaler=bool(CFG_A["use_scaler"]),
            class_weight=(
                None if CFG_A.get("class_weight") is None else str(CFG_A.get("class_weight"))
            ),
        )
        if model_family == "family_a"
        else None
    )

    record = build_record(
        tag=f"model_family_sweep_v1__{model_family}__ori_{orientation}",
        feature_name="hog",
        model_name=model_family,
        cv_type="GroupKFold(site)",
        n_splits=5,
        logreg_params=logreg_params,
        hog_params=hog_params,
        model_params=model_params,
        mean_log_loss=mean_log_loss,
        std_log_loss=standard_log_loss,
        fold_log_loss=scores,
        data_sig=data_sig,
        cv_checks=CVChecks(
            site_single_fold=True,
            no_site_overlap=True,
            random_state=42,
        ),
        fold_site_counts=fold_site_counts,
    )

    log_experiment(LOG_PATH, record.to_dict())
    print(
        f"[log] appended -> {LOG_PATH} | model={model_family}, ori={orientation}, "
        f"model_params={model_params}"
    )
