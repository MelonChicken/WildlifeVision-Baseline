from pathlib import Path

import numpy as np
import pandas as pd

from src.data.image_process import image_preprocess
from src.features.hog import extract_hog


def build_hog_features(
    df: pd.DataFrame,
    *,
    project_dir: Path,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
    orientations: int,
    block_norm: str,
) -> np.ndarray:
    x_list: list[np.ndarray] = []
    for image_path in df["filepath"]:
        x_img = image_preprocess(img_path=image_path, base_path=project_dir)
        hog_feature = extract_hog(
            x_img,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
            block_norm=block_norm,
        )
        x_list.append(hog_feature)

    X = np.vstack(x_list).astype(np.float32, copy=False)
    assert np.isfinite(X).all(), "HOG feature matrix contains NaN/Inf"
    return X
