from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config.global_variables import CLASS_COLS
from src.config.model_config import LogRegConfig
from src.config.paths import LOG_PATH
from src.data.image_process import image_preprocess
from src.data.make_table import make_train_table
from src.features.hog import (
    assert_tiled_feature_dim_quadrupled,
    extract_hog,
    extract_hog_tiled,
    get_hog_feature_dim,
)
from src.models.logreg import build_logreg_pipeline
from src.training.crossvalidation import (
    add_group_folds,
    check_no_site_overlap_between_train_valid,
    check_site_has_single_fold,
)
from src.training.evaluate import evaluate_by_fold
from src.training.experiment_logger import log_experiment


BASE_TAG = "hog_tune_v1"
FEATURE_NAME = "hog"
MODEL_NAME = "logreg"
CV_TYPE = "GroupKFold(site)"
FIXED_RANDOM_STATE = 42

BASELINE_HOG = {
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "orientations": 9,
    "block_norm": "L2-Hys",
}

FIXED_LOGREG_CFG = LogRegConfig(
    C=0.003,
    max_iter=6000,
    use_scaler=False,
    solver="lbfgs",
    random_state=42,
    class_weight="balanced",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HOG tuning experiments with fixed LogReg config.")
    parser.add_argument("--base_dir", type=str, default=".", help="Project base directory")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sanity-check sampling")
    return parser.parse_args()


def _make_y_labels(df) -> np.ndarray:
    row_sum = df[CLASS_COLS].sum(axis=1)
    assert (row_sum == 1).all(), "Each sample must have exactly one active class"
    y_idx = df[CLASS_COLS].to_numpy().argmax(axis=1)
    return np.array([CLASS_COLS[i] for i in y_idx], dtype=object)


def _sample_preprocess_and_hog_sanity(df, base_dir: Path, seed: int, n: int = 20) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)

    for i in idx:
        img_rel = df.iloc[int(i)]["filepath"]
        x = image_preprocess(img_path=img_rel, base_path=base_dir)

        assert x.shape == (128, 128), f"Unexpected preprocessed shape: {x.shape}"
        assert x.dtype == np.float32, f"Unexpected dtype: {x.dtype}"
        assert np.isfinite(x).all(), "Preprocessed image contains NaN/Inf"
        assert float(x.min()) >= 0.0, f"Preprocessed min out of range: {x.min()}"
        assert float(x.max()) <= 1.0, f"Preprocessed max out of range: {x.max()}"

        feat = extract_hog(
            x,
            pixels_per_cell=BASELINE_HOG["pixels_per_cell"],
            cells_per_block=BASELINE_HOG["cells_per_block"],
            orientations=BASELINE_HOG["orientations"],
            block_norm=BASELINE_HOG["block_norm"],
        )
        assert np.isfinite(feat).all(), "HOG feature contains NaN/Inf"


def _tiling_sanity_on_samples(df, base_dir: Path, seed: int, n: int = 20) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)

    baseline_dim = get_hog_feature_dim(
        image_shape_hw=(128, 128),
        pixels_per_cell=BASELINE_HOG["pixels_per_cell"],
        cells_per_block=BASELINE_HOG["cells_per_block"],
        orientations=BASELINE_HOG["orientations"],
        block_norm=BASELINE_HOG["block_norm"],
        tiled=False,
    )
    tiled_dim_expected = baseline_dim * 4

    for i in idx:
        img_rel = df.iloc[int(i)]["filepath"]
        x = image_preprocess(img_path=img_rel, base_path=base_dir)
        tiled_feat = extract_hog_tiled(
            x,
            tiles=(2, 2),
            pixels_per_cell=BASELINE_HOG["pixels_per_cell"],
            cells_per_block=BASELINE_HOG["cells_per_block"],
            orientations=BASELINE_HOG["orientations"],
            block_norm=BASELINE_HOG["block_norm"],
        )
        assert np.isfinite(tiled_feat).all(), "Tiled HOG feature contains NaN/Inf"
        assert tiled_feat.shape[0] == tiled_dim_expected, (
            f"Tiled dim mismatch: got={tiled_feat.shape[0]}, expected={tiled_dim_expected}"
        )


def _build_features(df, base_dir: Path, hog_params: dict, tiled: bool) -> np.ndarray:
    x_list: list[np.ndarray] = []

    for img_rel in df["filepath"]:
        x = image_preprocess(img_path=img_rel, base_path=base_dir)
        if tiled:
            feat = extract_hog_tiled(
                x,
                tiles=(2, 2),
                pixels_per_cell=hog_params["pixels_per_cell"],
                cells_per_block=hog_params["cells_per_block"],
                orientations=hog_params["orientations"],
                block_norm=hog_params["block_norm"],
            )
        else:
            feat = extract_hog(
                x,
                pixels_per_cell=hog_params["pixels_per_cell"],
                cells_per_block=hog_params["cells_per_block"],
                orientations=hog_params["orientations"],
                block_norm=hog_params["block_norm"],
            )

        if not np.isfinite(feat).all():
            raise ValueError(f"Non-finite HOG feature detected for file: {img_rel}")
        x_list.append(feat)

    X = np.vstack(x_list).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN/Inf")
    return X


def _build_record_payload(
    *,
    tag: str,
    n_splits: int,
    hog_params: dict,
    data_signature: dict,
    cv_checks: dict,
    fold_site_counts: dict,
    metrics: dict,
    tiled: bool,
    error: str | None = None,
) -> dict:
    params = {
        "logreg": {
            "C": float(FIXED_LOGREG_CFG.C),
            "max_iter": int(FIXED_LOGREG_CFG.max_iter),
            "use_scaler": bool(FIXED_LOGREG_CFG.use_scaler),
            "class_weight": FIXED_LOGREG_CFG.class_weight,
            "solver": FIXED_LOGREG_CFG.solver,
            "random_state": int(FIXED_LOGREG_CFG.random_state),
        },
        "hog": {
            "pixels_per_cell": [int(v) for v in hog_params["pixels_per_cell"]],
            "cells_per_block": [int(v) for v in hog_params["cells_per_block"]],
            "orientations": int(hog_params["orientations"]),
            "block_norm": str(hog_params["block_norm"]),
            "tiled": bool(tiled),
        },
    }
    if tiled:
        params["hog"]["tiles"] = [2, 2]
    if error is not None:
        params["error"] = error

    payload = {
        "tag": tag,
        "feature_name": FEATURE_NAME,
        "model_name": MODEL_NAME,
        "cv": {
            "type": CV_TYPE,
            "n_splits": int(n_splits),
        },
        "params": params,
        "metrics": metrics,
        "data_signature": data_signature,
        "cv_checks": cv_checks,
        "fold_site_counts": {str(int(k)): int(v) for k, v in fold_site_counts.items()},
    }
    if error is not None:
        payload["error"] = error
    return payload


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    df_train = make_train_table(autosave=False)
    df_train = add_group_folds(
        df_train,
        group_col="site",
        n_splits=int(args.n_splits),
        fold_col="fold",
        shuffle=True,
        autosave=False,
        random_state=FIXED_RANDOM_STATE,
    )

    check_site_has_single_fold(df_train, site_col="site", fold_col="fold")
    check_no_site_overlap_between_train_valid(df_train, site_col="site", fold_col="fold")

    fold = df_train["fold"].to_numpy()
    y = _make_y_labels(df_train)

    class_counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    data_signature = {
        "n_samples": int(len(y)),
        "n_sites": int(df_train["site"].nunique()),
        "class_counts": class_counts,
    }
    cv_checks = {
        "site_single_fold": True,
        "no_site_overlap": True,
        "random_state": FIXED_RANDOM_STATE,
    }
    fold_site_counts = {
        int(k): int(v)
        for k, v in df_train.groupby("fold")["site"].nunique().to_dict().items()
    }

    # sanity checks (before experiments)
    _sample_preprocess_and_hog_sanity(df_train, base_dir=base_dir, seed=int(args.seed), n=20)
    assert_tiled_feature_dim_quadrupled(
        image_shape_hw=(128, 128),
        pixels_per_cell=BASELINE_HOG["pixels_per_cell"],
        cells_per_block=BASELINE_HOG["cells_per_block"],
        orientations=BASELINE_HOG["orientations"],
        block_norm=BASELINE_HOG["block_norm"],
        tiles=(2, 2),
    )

    variants = [
        {"variant_name": "baseline", "hog": dict(BASELINE_HOG), "tiled": False},
        {"variant_name": "tiled_2x2", "hog": dict(BASELINE_HOG), "tiled": True},
        {
            "variant_name": "ppc_4x4",
            "hog": {**dict(BASELINE_HOG), "pixels_per_cell": (4, 4)},
            "tiled": False,
        },
        {
            "variant_name": "ppc_16x16",
            "hog": {**dict(BASELINE_HOG), "pixels_per_cell": (16, 16)},
            "tiled": False,
        },
        {
            "variant_name": "ori_6",
            "hog": {**dict(BASELINE_HOG), "orientations": 6},
            "tiled": False,
        },
        {
            "variant_name": "ori_12",
            "hog": {**dict(BASELINE_HOG), "orientations": 12},
            "tiled": False,
        },
        {
            "variant_name": "cpb_3x3",
            "hog": {**dict(BASELINE_HOG), "cells_per_block": (3, 3)},
            "tiled": False,
        },
    ]

    for variant in variants:
        variant_name = str(variant["variant_name"])
        tag = f"{BASE_TAG}__{variant_name}"
        hog_params = dict(variant["hog"])
        tiled = bool(variant["tiled"])

        try:
            if variant_name == "tiled_2x2":
                # sanity check right before tiled run
                _tiling_sanity_on_samples(df_train, base_dir=base_dir, seed=int(args.seed), n=20)

            X = _build_features(df_train, base_dir=base_dir, hog_params=hog_params, tiled=tiled)

            if variant_name == "tiled_2x2":
                # sanity check right after tiled feature build
                baseline_dim = get_hog_feature_dim(
                    image_shape_hw=(128, 128),
                    pixels_per_cell=BASELINE_HOG["pixels_per_cell"],
                    cells_per_block=BASELINE_HOG["cells_per_block"],
                    orientations=BASELINE_HOG["orientations"],
                    block_norm=BASELINE_HOG["block_norm"],
                    tiled=False,
                )
                assert X.shape[1] == baseline_dim * 4, (
                    f"tiled_2x2 feature dim mismatch: got={X.shape[1]}, expected={baseline_dim * 4}"
                )
                assert np.isfinite(X).all(), "tiled_2x2 feature matrix contains NaN/Inf"

            mean_ll, std_ll, fold_ll = evaluate_by_fold(
                X=X,
                y=y,
                fold=fold,
                build_model_fn=lambda: build_logreg_pipeline(FIXED_LOGREG_CFG),
            )
            metrics = {
                "mean_log_loss": float(mean_ll),
                "std_log_loss": float(std_ll),
                "fold_log_loss": [float(v) for v in fold_ll],
            }

            payload = _build_record_payload(
                tag=tag,
                n_splits=int(args.n_splits),
                hog_params=hog_params,
                data_signature=data_signature,
                cv_checks=cv_checks,
                fold_site_counts=fold_site_counts,
                metrics=metrics,
                tiled=tiled,
            )
            log_experiment(LOG_PATH, payload)
            print(f"[log] appended -> {LOG_PATH} | {variant_name}")

        except Exception as e:
            err_msg = str(e)
            print(f"[warn] variant failed: {variant_name} | {err_msg}")
            metrics = {
                "mean_log_loss": None,
                "std_log_loss": None,
                "fold_log_loss": [],
            }
            payload = _build_record_payload(
                tag=tag,
                n_splits=int(args.n_splits),
                hog_params=hog_params,
                data_signature=data_signature,
                cv_checks=cv_checks,
                fold_site_counts=fold_site_counts,
                metrics=metrics,
                tiled=tiled,
                error=err_msg,
            )
            log_experiment(LOG_PATH, payload)
            print(f"[log] appended failure -> {LOG_PATH} | {variant_name}")


if __name__ == "__main__":
    main()
