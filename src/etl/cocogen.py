import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import loguru
import numpy as np
import PIL.Image
import PIL.ImageOps
from detectron2.data import transforms as T
from joblib import Parallel, delayed
from shapely.geometry import Polygon
from tqdm import tqdm

from ..defs import CROPS_PATH, INDEX_HASHES_PATH, RAW_DATA_PATH
from .annotation import (
    BADLY_ANNOTATED_HASHES,
    CONDITION_CLASS_TO_LABEL,
    CONDITION_LABEL_TO_CLASS,
    CONDITIONS_RU2EN,
    CROWN_DESTRUCTION_LABEL_TO_CLASS,
    INVOLVEMENT_LABEL_TO_CLASS,
    IS_BUILDUP_LABEL_TO_CLASS,
    PBL_SEVERITY_LABEL_TO_CLASS,
    PBL_TYPE_LABEL_TO_CLASS,
    POST_MATERIAL_LABEL_TO_CLASS,
    SURFACES_LABEL_TO_CLASS,
    Example,
)
from .tags import resolve_tags
from .utils import NpEncoder, get_examples_lakefs, read_image


def prepare_coco_dataset(output_path: str | Path, crop_context_px: int = 20) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(INDEX_HASHES_PATH, "r") as f:
        annotated_data_hashes = json.load(f)
    for data in annotated_data_hashes.values():
        data["hashes"] = set(data["hashes"])

    examples = get_examples_lakefs(RAW_DATA_PATH)

    annotated_hashes = (
        annotated_data_hashes["annotated"]["hashes"] - BADLY_ANNOTATED_HASHES
    )
    val_hashes = annotated_data_hashes["val"]["hashes"] - BADLY_ANNOTATED_HASHES
    test_hashes = annotated_data_hashes["test"]["hashes"] - BADLY_ANNOTATED_HASHES
    test_pipelines_hashes = (
        annotated_data_hashes["pipelines_test_cases"]["hashes"] - BADLY_ANNOTATED_HASHES
    )
    train_hashes = annotated_hashes - val_hashes - test_hashes - test_pipelines_hashes

    train_examples = [ex for ex in examples if ex["image_hash"] in train_hashes]
    val_examples = [ex for ex in examples if ex["image_hash"] in val_hashes]
    test_examples = [ex for ex in examples if ex["image_hash"] in test_hashes]
    test_pipelines_examples = [
        ex for ex in examples if ex["image_hash"] in test_pipelines_hashes
    ]
    loguru.logger.info(f"Train examples: {len(train_examples)}")
    loguru.logger.info(f"Val examples: {len(val_examples)}")
    loguru.logger.info(f"Test examples: {len(test_examples)}")
    loguru.logger.info(f"Test pipelines examples: {len(test_pipelines_examples)}")

    tags_meta = {
        "is_buildup": IS_BUILDUP_LABEL_TO_CLASS,
        "post_material": POST_MATERIAL_LABEL_TO_CLASS,
        "involvement": INVOLVEMENT_LABEL_TO_CLASS,
        "pbl_severity": PBL_SEVERITY_LABEL_TO_CLASS,
        "pbl_type": PBL_TYPE_LABEL_TO_CLASS,
        "crown_destruction": CROWN_DESTRUCTION_LABEL_TO_CLASS,
        **{
            f"is_surface_{surface}": label_to_class
            for surface, label_to_class in SURFACES_LABEL_TO_CLASS.items()
        },
    }
    categories = []
    for name, label in CONDITION_CLASS_TO_LABEL.items():
        categories.append(
            {
                "id": label,
                "name": str(name),
                "supercategory": str(name),
                "tags": tags_meta,
            }
        )

    build_coco_dataset(
        train_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "train",
        annotation_file_output_path=output_path
        / "annotations"
        / "instances_train.json",
        crop_context_px=crop_context_px,
    )

    build_coco_dataset(
        val_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "val",
        annotation_file_output_path=output_path / "annotations" / "instances_val.json",
        start_image_id=100000,
        crop_context_px=crop_context_px,
    )

    build_coco_dataset(
        test_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "test",
        annotation_file_output_path=output_path / "annotations" / "instances_test.json",
        start_image_id=200000,
        crop_context_px=crop_context_px,
    )

    build_coco_dataset(
        test_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "test_orig",
        annotation_file_output_path=output_path
        / "annotations"
        / "instances_test_orig.json",
        start_image_id=300000,
        crop_context_px=None,
        short_edge_length=None,
    )

    build_coco_dataset(
        test_pipelines_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "test_pipelines",
        annotation_file_output_path=output_path
        / "annotations"
        / "instances_test_pipelines.json",
        start_image_id=400000,
        crop_context_px=crop_context_px,
    )

    build_coco_dataset(
        test_pipelines_examples,
        categories=categories,
        image_input_path=RAW_DATA_PATH / "data",
        image_output_path=output_path / "test_pipelines_orig",
        annotation_file_output_path=output_path
        / "annotations"
        / "instances_test_pipelines_orig.json",
        start_image_id=500000,
        crop_context_px=None,
        short_edge_length=None,
    )


def build_coco_dataset(
    examples: list[Example],
    categories: list[dict],
    image_input_path: Path,
    image_output_path: Path,
    annotation_file_output_path: Path,
    short_edge_length: int | None = 1024,
    long_edge_max_length: int = 2048,
    crop_context_px: int | None = None,
    start_image_id: int = 1,
    n_jobs: int = 16,
) -> None:
    image_output_path.mkdir(parents=True, exist_ok=True)
    annotation_file_output_path.parent.mkdir(parents=True, exist_ok=True)

    loguru.logger.info(f"Building dataset {image_output_path}")

    if short_edge_length is not None:
        loguru.logger.info(f"Resizing image's shortest side to {short_edge_length}")

    if crop_context_px is not None:
        assert CROPS_PATH.exists()
        with open(CROPS_PATH, "r") as f:
            crops = json.load(f)
        for image_hash, crop in crops.items():
            ymin, xmin, ymax, xmax = crop
            crops[image_hash] = (
                ymin - crop_context_px,
                xmin - crop_context_px,
                ymax + crop_context_px,
                xmax + crop_context_px,
            )
    else:
        crops = {}

    out = Parallel(n_jobs=n_jobs)(
        delayed(process_example)(
            example=example,
            image_input_path=image_input_path,
            image_output_path=image_output_path,
            short_edge_length=short_edge_length,
            long_edge_max_length=long_edge_max_length,
            crop=crops.get(example["image_hash"]),
        )
        for example in tqdm(examples)
    )

    images = []
    annotations = []
    image_id = start_image_id
    annotation_id = 1
    stats = defaultdict(int)
    for image, annos in out:
        if image is None:
            continue
        image["id"] = image_id
        images.append(image)
        for anno in annos:
            anno["id"] = annotation_id
            anno["image_id"] = image_id
            annotation_id += 1
            annotations.append(anno)

            stats["total_objects"] += 1
            condition = CONDITION_LABEL_TO_CLASS[anno["category_id"]]
            stats[f"Condition/{condition}"] += 1
            if not anno["is_mask_annotated"]:
                stats[f"Condition/{condition}/mask not annotated"] += 1

            is_buildup = IS_BUILDUP_LABEL_TO_CLASS.get(anno["is_buildup"])
            if is_buildup is not None:
                stats[f"Is Build-up/{is_buildup}"] += 1

            post_material = POST_MATERIAL_LABEL_TO_CLASS.get(anno["post_material"])
            if post_material is not None:
                stats[f"Post Material/{post_material}"] += 1

            involvement = INVOLVEMENT_LABEL_TO_CLASS.get(anno["involvement"])
            if involvement is not None:
                stats[f"Involvement/{involvement}"] += 1

            pbl_severity = PBL_SEVERITY_LABEL_TO_CLASS.get(anno["pbl_severity"])
            if pbl_severity is not None:
                stats[f"PBL Severity/{pbl_severity}"] += 1

            pbl_type = PBL_TYPE_LABEL_TO_CLASS.get(anno["pbl_type"])
            if pbl_type is not None:
                stats[f"PBL Type/{pbl_type}"] += 1

            crown_destruction = CROWN_DESTRUCTION_LABEL_TO_CLASS.get(
                anno["crown_destruction"]
            )
            if crown_destruction is not None:
                stats[f"Crown Destruction/{crown_destruction}"] += 1

            for surface, surface_label_to_class in SURFACES_LABEL_TO_CLASS.items():
                surface_class = surface_label_to_class.get(
                    anno[f"is_surface_{surface}"]
                )
                if surface_class is not None:
                    stats[f"Surface/{surface_class}"] += 1

        image_id += 1

    loguru.logger.info("Dataset Stats:")
    for stat_name in sorted(stats):
        loguru.logger.info(f"{stat_name}: {stats[stat_name]}")

    data = {
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    loguru.logger.info(
        f"{annotation_file_output_path=} ",
    )
    with (annotation_file_output_path).open("w") as f:
        json.dump(data, f, cls=NpEncoder)


def process_example(
    example: Example,
    image_input_path: Path,
    image_output_path: Path,
    short_edge_length: int | None = 768,
    long_edge_max_length: int = 1333,
    ignore_label: int = -100,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    img_out_path = image_output_path / example["image_path"]
    img_out_path.parent.mkdir(parents=True, exist_ok=True)

    img_path = image_input_path / example["image_path"]

    if not img_path.exists():
        loguru.logger.warning(f"Image {img_path} doesn't exist, skipping...")
        return None, None

    image = read_image(img_path, bgr2rgb=False)
    if image is None:
        return None, None

    if image.ndim == 3:
        image = image[:, :, 0]

    if image.dtype != np.uint8:
        ratio = image.max() / 255
        image = (image / ratio).astype(np.uint8)

    if crop is not None:
        ymin_wf, xmin_wf, ymax_wf, xmax_wf = crop
        ymin_wf = max(0, ymin_wf)
        xmin_wf = max(0, xmin_wf)
        ymax_wf = min(image.shape[0], ymax_wf)
        xmax_wf = min(image.shape[1], xmax_wf)
        image = image[ymin_wf:ymax_wf, xmin_wf:xmax_wf]

    if short_edge_length is not None:
        transform = T.ResizeShortestEdge(
            short_edge_length, long_edge_max_length, "choice"
        )
        transform = transform.get_transform(image)
        image = transform.apply_image(image)
    else:
        transform = T.NoOpTransform()

    image = PIL.Image.fromarray(image)
    image.save(img_out_path)

    width, height = image.size

    image = {
        "file_name": str(img_out_path.relative_to(image_output_path)),
        "height": height,
        "width": width,
    }

    annos = []
    for condition_ru, condition_en in CONDITIONS_RU2EN.items():
        if (condition_label := CONDITION_CLASS_TO_LABEL.get(condition_en)) is None:
            continue

        objects = [obj for obj in example["objects"] if obj["label"] == condition_ru]

        for object_ in objects:
            anno: dict[str, Any] = {
                "iscrowd": 0,
                "category_id": condition_label,
            }
            assert isinstance(object_["points"], list)

            points = np.array(object_["points"])
            if crop is not None:
                points -= np.array([xmin_wf, ymin_wf])
                if (points < 0).all():
                    continue
                points = points.clip(min=0)
            object_["points"] = transform.apply_coords(points).tolist()

            if object_["shape"] in ("polygon", "points"):
                if len(object_["points"]) < 3:
                    continue
                poly = Polygon(object_["points"])
                xmin, ymin, xmax, ymax = cast(list[int], poly.bounds)
                anno["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
                anno["area"] = poly.area
                anno["segmentation"] = [np.array(poly.exterior.coords).ravel().tolist()]
                anno["is_mask_annotated"] = True
                # if len(object_["points"]) == 4:
                #     anno["is_mask_annotated"] = False
                # else:
                #     anno["is_mask_annotated"] = True
            elif object_["shape"] == "bbox":
                [[xmin, ymin], [xmax, ymax]] = object_["points"]
                anno["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
                anno["area"] = (xmax - xmin) * (ymax - ymin)
                anno["segmentation"] = [
                    [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
                ]
                anno["is_mask_annotated"] = False

            elif object_["shape"] == "line":
                continue

            else:
                loguru.logger.warning(f"Unknown shape: {object_['shape']}")
                continue

            tags = resolve_tags(
                condition_en, object_["tags"], ignore_label=ignore_label
            )
            anno.update(tags)

            annos.append(anno)

    return image, annos
