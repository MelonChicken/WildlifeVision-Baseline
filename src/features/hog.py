from pathlib import Path

import numpy as np
import pandas as pd
from skimage.feature import hog

from src.config.global_variables import CLASS_COLS
from src.data.image_process import image_preprocess


def extract_hog(
    x: np.ndarray,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    orientations: int = 9,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """
    Extract HOG feature vector from a grayscale image.
    Params:
      x: Grayscale image array (H, W). float32 recommended.
      pixels_per_cell: Cell size in pixels (H, W).
      cells_per_block: Block size in cells (H, W).
      orientations: Number of orientation bins.
      block_norm: Block normalization method (e.g., "L2-Hys").
    Returns:
      1D feature vector (n_features,), float32
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (H, W). Got shape={x.shape}")

    feat = hog(
        x,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    ).astype(np.float32)

    return feat


def sanity_check_hog(
        df: pd.DataFrame,
        project_root: Path,
        n=20,
        seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)

    lens = set()
    shapes = set()
    failed = []

    for i in idx:
        filepath = df.iloc[i]["filepath"]
        try:
            x = image_preprocess(img_path=filepath, base_path=project_root)
            shapes.add(x.shape)

            feat = extract_hog(x)
            lens.add(feat.shape[0])

        except Exception as e:
            failed.append((str(filepath), str(e)))

    return {
        "n_checked": len(idx),
        "unique_image_shapes": sorted(list(shapes)),
        "unique_feature_lengths": sorted(list(lens)),
        "n_failed": len(failed),
        "failed_examples": failed[:5],
    }

def load_or_build_train_hog_cache(
    df: pd.DataFrame,
    project_dir: Path,
    artifacts_dir: Path,
    prefix: str = "train_hog",
):
    """
    Build HOG features once and cache them as .npy files.
    Cache files:
      - {prefix}_X.npy
      - {prefix}_y.npy
      - {prefix}_fold.npy

    Assumptions:
      - df has columns: filepath, label, fold
      - df order is stable between runs (row alignment relies on order)
    """
    feature_dir = artifacts_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    X_path = feature_dir / f"{prefix}_X.npy"
    y_path = feature_dir / f"{prefix}_y.npy"
    fold_path = feature_dir / f"{prefix}_fold.npy"
    filepath_path = feature_dir / f"{prefix}_filepath.npy"

    expected_fp = df["filepath"].to_numpy()

    # load if exists
    if X_path.exists() and y_path.exists() and fold_path.exists():
        print(f"[cache] loading: {X_path.name}, {y_path.name}, {fold_path.name}")
        X = np.load(X_path)
        y = np.load(y_path, allow_pickle=True)
        fold = np.load(fold_path)
        assert X.ndim == 2 and y.ndim == 1 and fold.ndim == 1
        assert len(X) == len(y) == len(fold)
        return X, y, fold

    # build + save
    print("[cache] building HOG features (no cache found)")
    X_list = []

    for image_path in df["filepath"]:
        x_img = image_preprocess(
            img_path=image_path,
            base_path=project_dir
        )
        hog_feature = extract_hog(x_img)
        X_list.append(hog_feature)
    # convert X_list as 2-Dimension vector (number of samples, 8100)
    X = np.vstack(X_list).astype(np.float32, copy=False)

    # make 2-Dimension vector to save labels (answer y)
    row_sum = df[CLASS_COLS].sum(axis=1)
    assert (row_sum == 1).all(), "원-핫 라벨 행 합이 1이 아닌 샘플이 있습니다."

    y_idx = df[CLASS_COLS].to_numpy().argmax(axis=1)
    y = np.array([CLASS_COLS[i] for i in y_idx], dtype=object)

    # make 2-Dimension vector to save site and fold
    fold = df["fold"].to_numpy()

    # minimal sanity checks
    assert len(X) == len(y) == len(fold), "X/y/fold 길이가 다릅니다."
    assert X.ndim == 2 and y.ndim == 1 and fold.ndim == 1

    np.save(X_path, X)
    np.save(y_path, y)
    np.save(fold_path, fold)
    np.save(filepath_path, expected_fp, allow_pickle=True)
    print(f"[cache] saved: {X_path}, {y_path}, {fold_path}")

    return X, y, fold

from pathlib import Path
import numpy as np
import pandas as pd

def load_or_build_test_hog_cache(
    df: pd.DataFrame,
    project_dir: Path,
    artifacts_dir: Path,
    prefix: str = "test_hog",
):
    """
    Build HOG features for test once and cache them as .npy files.
    Cache files:
      - {prefix}_X.npy
      - {prefix}_id.npy
      - {prefix}_filepath.npy

    Assumptions:
      - df has columns: filepath
      - df has id either as index name 'id' or column 'id'
      - df order is stable between runs (row alignment relies on order)
    """
    feature_dir = artifacts_dir / "features"

    X_path = feature_dir / f"{prefix}_X.npy"
    id_path = feature_dir / f"{prefix}_id.npy"
    filepath_path = feature_dir / f"{prefix}_filepath.npy"

    # id 확보: (1) index가 id이면 index 사용, (2) 아니면 'id' 컬럼 사용
    if df.index.name == "id":
        ids = df.index.to_numpy()
    elif "id" in df.columns:
        ids = df["id"].to_numpy()
    else:
        raise ValueError("test df must have id as index(name='id') or a column named 'id'.")

    expected_fp = df["filepath"].to_numpy()

    # load if exists (+ 입력 파일 목록 동일성 체크)
    if X_path.exists() and id_path.exists() and filepath_path.exists():
        print(f"[cache] loading: {X_path.name}, {id_path.name}, {filepath_path.name}")
        X = np.load(X_path)
        cached_ids = np.load(id_path, allow_pickle=True)
        cached_fp = np.load(filepath_path, allow_pickle=True)

        assert X.ndim == 2 and cached_ids.ndim == 1 and cached_fp.ndim == 1
        assert len(X) == len(cached_ids) == len(cached_fp)

        # 캐시가 현재 df와 같은 순서/내용인지 확인
        if not (np.array_equal(cached_ids, ids) and np.array_equal(cached_fp, expected_fp)):
            raise ValueError(
                "Cached test HOG does not match current test table order/paths. "
                "Delete cache files or change prefix."
            )

        return X, cached_ids

    # build + save
    print("[cache] building TEST HOG features (no cache found)")
    X_list = []
    for image_path in df["filepath"]:
        x_img = image_preprocess(img_path=image_path, base_path=project_dir)
        hog_feature = extract_hog(x_img)
        X_list.append(hog_feature)

    X = np.vstack(X_list).astype(np.float32, copy=False)

    # minimal sanity checks
    assert X.ndim == 2
    assert len(X) == len(ids), "X와 id 길이가 다릅니다."

    np.save(X_path, X)
    np.save(id_path, ids, allow_pickle=True)
    np.save(filepath_path, expected_fp, allow_pickle=True)
    print(f"[cache] saved: {X_path}, {id_path}, {filepath_path}")

    return X, ids
