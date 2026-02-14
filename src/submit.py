# src/submit.py (핵심만 올바르게 정리한 버전)

from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd

from src.config.model_config import LogRegConfig
from src.config.paths import DATA_DIR, SUBMISSIONS_DIR, PROJECT_ROOT, ARTIFACTS_DIR
from src.data.make_table import make_train_table, make_test_table
from src.features.hog import load_or_build_train_hog_cache, load_or_build_test_hog_cache
from src.models.logreg import build_logreg_pipeline

SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"


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


def save_submission(df_sub: pd.DataFrame, tag: str = "hog_logreg_v1") -> str:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"sub_{ts}_{tag}.csv"
    df_sub.to_csv(out_path, index=True)
    return str(out_path)


def main():
    # 0) 제출 템플릿 로드
    df_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    class_cols = list(df_format.columns)

    # 1) 테이블 로드
    df_train = make_train_table(autosave=True)
    df_test = make_test_table(autosave=True)  # index=id, columns: filepath, site

    # 2) train HOG 캐시
    X_train, y_train, _ = load_or_build_train_hog_cache(
        df=df_train,
        project_dir=PROJECT_ROOT,
        artifacts_dir=ARTIFACTS_DIR,
        prefix="train_hog",
    )

    # 3) test HOG 캐시 (메타데이터로부터 이미지 읽어 HOG 추출)
    # 반환은 (X_test, test_ids) 형태로 맞춰두는 게 안전
    X_test, test_ids = load_or_build_test_hog_cache(
        df=df_test,
        project_dir=PROJECT_ROOT,
        artifacts_dir=ARTIFACTS_DIR,
        prefix="test_hog",
    )

    # 4) 모델 학습(전체 train)
    cfg = LogRegConfig(
        C=0.1,
        max_iter=2000,
        use_scaler=False,
        solver="lbfgs",
        random_state=42,
    )
    pipe = build_logreg_pipeline(cfg)
    pipe.fit(X_train, y_train)

    # 5) 확률 예측 + 컬럼 정렬
    proba = pipe.predict_proba(X_test)
    model_classes = pipe.named_steps["model"].classes_.tolist()

    if set(model_classes) != set(class_cols):
        raise ValueError(f"Class set mismatch. model={model_classes}, format={class_cols}")

    reorder_idx = [model_classes.index(c) for c in class_cols]
    proba = proba[:, reorder_idx]

    # 6) submission_format 순서로 정렬해서 제출 DF 구성
    df_proba = pd.DataFrame(proba, index=test_ids, columns=class_cols)
    df_sub = df_proba.loc[df_format.index]

    # 7) 검증 + 저장
    validate_submission(df_sub, df_format)
    out_path = save_submission(df_sub, tag="hog_logreg_v1")
    print(f"[OK] Saved submission: {out_path}")


if __name__ == "__main__":
    main()