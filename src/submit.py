from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.model_config import LogRegConfig
from src.config.paths import DATA_DIR, SUBMISSIONS_DIR, PROJECT_ROOT, LOG_PATH
from src.data.image_process import image_preprocess
from src.data.make_table import make_train_table, make_test_table
from src.features.hog import extract_hog, extract_hog_tiled
from src.models.logreg import build_logreg_pipeline
from src.config.global_variables import CLASS_COLS

SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build submission by reproducing a logged experiment (run_id)."
    )
    parser.add_argument("--run_id", type=str, required=True, help="Target run_id in experiments.jsonl")
    parser.add_argument("--log_path", type=str, default=str(LOG_PATH), help="Path to experiments.jsonl")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root for resolving image paths",
    )
    return parser.parse_args()


def validate_submission(df_sub: pd.DataFrame, df_format: pd.DataFrame) -> None:
    if len(df_sub) != len(df_format):
        raise ValueError(f"Row mismatch: sub={len(df_sub)}, format={len(df_format)}")
    if not df_sub.index.equals(df_format.index):
        raise ValueError("Index(id) order mismatch with submission_format.csv")
    if list(df_sub.columns) != list(df_format.columns):
        raise ValueError("Class columns mismatch or order mismatch with submission_format.csv")

    arr = df_sub.to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError("Found NaN/inf in submission")
    row_sum = arr.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-6):
        max_err = float(np.max(np.abs(row_sum - 1.0)))
        raise ValueError(f"Row probability sums not 1.0 (max abs err={max_err})")


def save_submission(df_sub: pd.DataFrame, tag: str) -> str:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"sub_{ts}_{tag}.csv"
    df_sub.to_csv(out_path, index=True)
    return str(out_path)


def _find_experiment_by_run_id(log_path: Path, run_id: str) -> dict:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing experiments log: {log_path}")

    found = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("run_id")) == run_id:
                found = rec

    if found is None:
        raise ValueError(f"run_id not found in log: {run_id}")

    return found


def _to_hog_params(exp: dict) -> tuple[dict, bool]:
    params = exp.get("params", {})
    hog_params = params.get("hog", {})

    required = ["pixels_per_cell", "cells_per_block", "orientations", "block_norm"]
    missing = [k for k in required if k not in hog_params]
    if missing:
        raise ValueError(f"Missing params.hog keys in experiment log: {missing}")

    converted = {
        "pixels_per_cell": tuple(int(v) for v in hog_params["pixels_per_cell"]),
        "cells_per_block": tuple(int(v) for v in hog_params["cells_per_block"]),
        "orientations": int(hog_params["orientations"]),
        "block_norm": str(hog_params["block_norm"]),
    }
    tiled = bool(hog_params.get("tiled", False))
    return converted, tiled


def _to_logreg_cfg(exp: dict) -> LogRegConfig:
    params = exp.get("params", {})
    logreg = params.get("logreg", {})

    required = ["C", "max_iter", "use_scaler", "class_weight"]
    missing = [k for k in required if k not in logreg]
    if missing:
        raise ValueError(f"Missing params.logreg keys in experiment log: {missing}")

    return LogRegConfig(
        C=float(logreg["C"]),
        max_iter=int(logreg["max_iter"]),
        use_scaler=bool(logreg["use_scaler"]),
        solver=str(logreg.get("solver", "lbfgs")),
        random_state=int(logreg.get("random_state", 42)),
        class_weight=logreg["class_weight"],
    )


def _labels_from_train_table(df_train: pd.DataFrame) -> np.ndarray:
    row_sum = df_train[CLASS_COLS].sum(axis=1)
    assert (row_sum == 1).all(), "Each sample must have exactly one active class"
    y_idx = df_train[CLASS_COLS].to_numpy().argmax(axis=1)
    return np.array([CLASS_COLS[i] for i in y_idx], dtype=object)


def _build_hog_matrix(
    filepaths: pd.Series,
    *,
    base_dir: Path,
    hog_params: dict,
    tiled: bool,
) -> np.ndarray:
    feature_list: list[np.ndarray] = []

    for rel_path in filepaths:
        x = image_preprocess(img_path=rel_path, base_path=base_dir)
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
            raise ValueError(f"Non-finite feature detected for image: {rel_path}")
        feature_list.append(feat)

    X = np.vstack(feature_list).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN/Inf")
    return X


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip()
    log_path = Path(args.log_path)
    base_dir = Path(args.base_dir).resolve()

    exp = _find_experiment_by_run_id(log_path=log_path, run_id=run_id)
    if exp.get("error"):
        raise ValueError(f"Selected run_id has error in log and cannot be reproduced: {exp['error']}")

    if exp.get("feature_name") != "hog" or exp.get("model_name") != "logreg":
        raise ValueError(
            f"Unsupported experiment type. feature_name={exp.get('feature_name')}, model_name={exp.get('model_name')}"
        )

    hog_params, tiled = _to_hog_params(exp)
    cfg = _to_logreg_cfg(exp)

    print(f"[info] selected run_id={run_id}, tag={exp.get('tag')}")
    print(f"[info] hog_params={hog_params}, tiled={tiled}")
    print(
        "[info] logreg_cfg="
        f"C={cfg.C}, max_iter={cfg.max_iter}, use_scaler={cfg.use_scaler}, "
        f"solver={cfg.solver}, random_state={cfg.random_state}, class_weight={cfg.class_weight}"
    )

    df_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    class_cols = list(df_format.columns)

    df_train = make_train_table(autosave=False)
    df_test = make_test_table(autosave=False)

    y_train = _labels_from_train_table(df_train)

    print("[info] building train HOG features from selected experiment params")
    X_train = _build_hog_matrix(
        df_train["filepath"],
        base_dir=base_dir,
        hog_params=hog_params,
        tiled=tiled,
    )

    print("[info] building test HOG features from selected experiment params")
    X_test = _build_hog_matrix(
        df_test["filepath"],
        base_dir=base_dir,
        hog_params=hog_params,
        tiled=tiled,
    )

    pipe = build_logreg_pipeline(cfg)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)
    model_classes = pipe.named_steps["model"].classes_.tolist()

    if set(model_classes) != set(class_cols):
        raise ValueError(f"Class set mismatch. model={model_classes}, format={class_cols}")

    reorder_idx = [model_classes.index(c) for c in class_cols]
    proba = proba[:, reorder_idx]

    test_ids = df_test.index.to_numpy()
    df_proba = pd.DataFrame(proba, index=test_ids, columns=class_cols)
    df_sub = df_proba.loc[df_format.index]

    validate_submission(df_sub, df_format)

    tag = str(exp.get("tag", "logreg_hog"))
    out_path = save_submission(df_sub, tag=f"{tag}__{run_id}")
    print(f"[OK] Saved submission: {out_path}")


if __name__ == "__main__":
    main()
