import json

import cv2
import loguru
import pandas as pd
import PIL.Image
from pipelines.graphs_2d.pano.graph_spec import build_graph_pano
from pipelines.graphs_2d.pano.types import DetectedToothPano
from pipelines.pipelib.bbox import Box2D
from pipelines.workflows.compute import LinearComputeExecutor
from tqdm import tqdm

from src.defs import CROPS_PATH, INDEX_HASHES_PATH, RAW_DATA_PATH, ROOT
from src.etl.utils import get_examples_lakefs, read_image


def main():

    with open(INDEX_HASHES_PATH, "r") as f:
        annotated_data_hashes = json.load(f)

    examples = get_examples_lakefs(RAW_DATA_PATH)
    annotated_hashes = set(annotated_data_hashes["annotated"]["hashes"])
    annotated_examples = [ex for ex in examples if ex["image_hash"] in annotated_hashes]

    executor = LinearComputeExecutor(
        spec=build_graph_pano(localizer_version="v2024"),
        verbose=False,
    )

    if CROPS_PATH.exists():
        with open(CROPS_PATH, "r") as f:
            crops = json.load(f)
    else:
        crops = {}

    examples_to_process = [
        ex for ex in annotated_examples if ex["image_hash"] not in crops
    ]

    for example in tqdm(examples_to_process):
        img_path = RAW_DATA_PATH / "data" / example["image_path"]
        image = read_image(img_path, bgr2rgb=False)
        if image is None:
            continue
        executor["input"] = PIL.Image.fromarray(image)
        teeth_localization = executor["teeth_localization"]
        crop = resolve_crop(teeth_localization, *image.shape[:2])
        crops[example["image_hash"]] = [
            int(crop.ymin),
            int(crop.xmin),
            int(crop.ymax),
            int(crop.xmax),
        ]

    with open(CROPS_PATH, "w") as f:
        json.dump(crops, f)


def resolve_crop(
    teeth_localization: list[DetectedToothPano], image_height: int, image_width: int
) -> Box2D:
    if len(teeth_localization) == 0:
        return Box2D(
            ymin=0,
            xmin=0,
            ymax=image_height,
            xmax=image_width,
            shape=(image_height, image_width),
        )

    crop = teeth_localization[0]["bbox"]
    for tooth in teeth_localization:
        crop = crop.extend(tooth["bbox"])

    return crop


if __name__ == "__main__":
    main()
