import json
from pathlib import Path
from typing import Union, cast

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .annotation import CONDITIONS_RU2EN, Example


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_examples_lakefs(lakefs_path: Path, n_jobs: int = -1) -> list[Example]:
    assert (lakefs_path / "data").exists()

    examples = cast(
        list[Example],
        Parallel(n_jobs=n_jobs)(
            delayed(read_example)(fpath)
            for fpath in tqdm(
                (lakefs_path / "data").glob("**/*.json"), desc="Reading examples..."
            )
        ),
    )

    return examples


def read_example(fpath: Path) -> Example:
    with open(fpath, "r") as f:
        example = json.load(f)
    return example


def read_image(
    path: Union[Path, str], bgr2rgb: bool = True, read_grayscale: bool = False
) -> np.ndarray | None:
    """
    Return a RGB image

    Args:
        path (Path): path to image

    Returns:
        np.ndarray: H x W x 3 array
    """
    flag = cv2.IMREAD_GRAYSCALE if read_grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is not None and bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def translate_annotated_classes(annotated_classes_ru: list[str]) -> list[str]:
    out = []
    for condition_ru in annotated_classes_ru:
        if (condition_en := CONDITIONS_RU2EN.get(condition_ru)) is not None:
            out.append(condition_en)
    return list(set(out))
