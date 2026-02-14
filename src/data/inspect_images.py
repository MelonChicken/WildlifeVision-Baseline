from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from src.config.paths import PROJECT_ROOT


def collect_image_sizes(
        filepaths: List[str],
        base_dir: str = None
):
    """
    Collect image width/height without fully loading pixels.
    :param filepaths:
    :param base_dir: if filepaths is relative path, the base_dir will be needed to indicate the absolute path.
    :return:
      - df: columns = [path, w, h, resolution, pixels, aspect, aspect_round]
      - summary
    """
    if not (base_dir is None):
        base_dir = Path(base_dir)
    else:
        base_dir = PROJECT_ROOT

    rows = []
    # if the image load is failed
    failed = 0

    for filepath in filepaths:
        path = base_dir / filepath
        if not path.exists():
            failed += 1
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
            rows.append({"path": str(filepath), "w": w, "h": h})
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
            failed += 1

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise Exception("No images found")

    df['resolution'] = df["w"].astype(str) + "x" + df["h"].astype(str)
    df['pixels'] = df['w'] * df['h']
    df['aspect'] = df['w'] / df['h']
    df['aspect_round'] = df['aspect'].round(2)

    summary = {
        "n_total": len(filepaths),
        "n_ok": len(df),
        "n_failed": failed,
        "min_w_h": (int(df["w"].min()), int(df["h"].min())),
        "max_w_h": (int(df["w"].max()), int(df["h"].max())),
        "min_pixels": int(df["pixels"].min()),
        "max_pixels": int(df["pixels"].max()),
        "unique_resolutions": int(df["resolution"].nunique()),
        "unique_aspect_rounded": int(df["aspect_round"].nunique()),
    }
    return df, summary


def plot_image_size_distributions(
        df_sizes : pd.DataFrame,
        top_k=15,
        output_dir: Path = None,
):
    """
    Plot distributions: width, height, pixels(log), aspect ratio, and top resolutions.
    """
    # 1) Width / Height histogram
    plt.figure()
    plt.hist(df_sizes["w"], bins=30)
    plt.title("Width distribution")
    plt.xlabel("Width (px)")
    plt.ylabel("Count")
    plt.savefig(output_dir / "width_dist.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.hist(df_sizes["h"], bins=30)
    plt.title("Height distribution")
    plt.xlabel("Height (px)")
    plt.ylabel("Count")
    plt.savefig(output_dir / "height_dist.png", dpi=200, bbox_inches="tight")

    # 2) Pixels distribution (log scale is usually more readable)
    pixels = df_sizes["pixels"].astype(float).values
    plt.figure()
    plt.hist(np.log10(pixels + 1), bins=30)
    plt.title("Pixel count distribution (log10)")
    plt.xlabel("log10(w*h + 1)")
    plt.ylabel("Count")
    plt.savefig(output_dir / "pixels_dist.png", dpi=200, bbox_inches="tight")

    # 3) Aspect ratio histogram
    plt.figure()
    plt.hist(df_sizes["aspect"], bins=30)
    plt.title("Aspect ratio distribution (w/h)")
    plt.xlabel("Aspect ratio")
    plt.ylabel("Count")
    plt.savefig(output_dir / "aspect_dist.png", dpi=200, bbox_inches="tight")

    # 4) Top resolutions bar chart
    top_res = df_sizes["resolution"].value_counts().head(top_k)
    plt.figure(figsize=(10, 5))
    plt.bar(top_res.index.astype(str), top_res.values)
    plt.title(f"Top {top_k} resolutions")
    plt.xlabel("Resolution (w x h)")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"top{top_k}_resolutions_dist.png", dpi=200, bbox_inches="tight")
