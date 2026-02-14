from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import random

def image_preprocess(
    img_path: str | Path,
    base_path: str | Path=None,
    out_size=(128, 128),
    normalize=True,
    pad_value=0,
) -> np.ndarray:
    """
    Load image as grayscale and resize with aspect ratio preserved (letterbox).
    :param base_path: base path to project
    :param img_path: path to image
    :param out_size: output image size (H, W)
    :param normalize: whether to normalize image
    :param pad_value: padding value
    :return: resized image (H, W) float32
    """
    image_path = base_path / img_path if base_path else img_path

    with Image.open(image_path) as img:
        # convert into grayscale
        img = img.convert("L")

        output_height, output_width = out_size
        width, height = img.size

        scale = min(output_width / width, output_height / height)

        # resize image
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image_resized = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

        # add padding to ensure that the output image keeps its aspect and satisfies the output size
        canvas = Image.new("L", (output_width, output_height), color=pad_value)
        left = (output_width - new_width) // 2
        top = (output_height - new_height) // 2
        canvas.paste(image_resized, (left, top))

        x = np.asarray(canvas, dtype=np.float32)

    if normalize:
        x /= 255.0

    return x

def image_sanity_check(
        file_paths: List[str],
        base_dir: Path,
        n: int = 20,
        seed: int = 42,
):
    """
        Check shape consistency and read failures for letterboxed preprocessing.
        :param file_paths: list of file paths
        :param base_dir: base directory
        :param n: number of images to check
        :param seed: random seed
        """
    random.seed(seed)
    samples = random.sample(sorted(file_paths), k = min(n, len(file_paths)))
    shapes  = set()
    failed = []

    for sample in samples:
        path = base_dir / sample if base_dir else Path(sample)
        try:
            img = image_preprocess(img_path=path, out_size=(128, 128))
            shapes.add(img.shape)
        except Exception as e:
            failed.append((str(sample), str(e)))

    return {
        "n_checked": len(samples),
        "unique_shapes": sorted(list(shapes)),
        "n_failed": len(failed),
        "failed_examples": failed[:5],
    }