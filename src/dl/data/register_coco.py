from __future__ import annotations

from pathlib import Path

from detectron2.data import DatasetCatalog

from ...defs import PROCESSED_DATA_PATH
from .utils import load_coco_json


def register_available_datasets(
    datasets: list[str] | None = None, tag_names: list[str] | None = None
) -> None:
    if datasets is None:
        datasets = ["coco"]

    for dataset_name in datasets:
        dataset_path = PROCESSED_DATA_PATH / dataset_name
        register_coco_instances_w_extra_keys(
            f"{dataset_name}-train",
            dataset_path / "annotations/instances_train.json",
            dataset_path / "train",
            extra_annotation_keys=["is_mask_annotated"],
            tag_names=tag_names,
        )
        register_coco_instances_w_extra_keys(
            f"{dataset_name}-val",
            dataset_path / "annotations/instances_val.json",
            dataset_path / "val",
            extra_annotation_keys=["is_mask_annotated"],
            tag_names=tag_names,
        )
        register_coco_instances_w_extra_keys(
            f"{dataset_name}-test",
            dataset_path / "annotations/instances_test.json",
            dataset_path / "test",
            extra_annotation_keys=["is_mask_annotated"],
            tag_names=tag_names,
        )
        register_coco_instances_w_extra_keys(
            f"{dataset_name}-test-orig",
            dataset_path / "annotations/instances_test_orig.json",
            dataset_path / "test_orig",
            extra_annotation_keys=["is_mask_annotated"],
            tag_names=tag_names,
        )
        register_coco_instances_w_extra_keys(
            f"{dataset_name}-test-pipelines-orig",
            dataset_path / "annotations/instances_test_pipelines_orig.json",
            dataset_path / "test_pipelines_orig",
            extra_annotation_keys=["is_mask_annotated"],
            tag_names=tag_names,
        )


def register_coco_instances_w_extra_keys(
    dataset_name: str,
    json_file: Path,
    image_root: Path,
    extra_annotation_keys: list[str] | None = None,
    tag_names: list[str] | None = None,
) -> None:
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(
            str(json_file),
            str(image_root),
            dataset_name,
            extra_annotation_keys=extra_annotation_keys,
            tag_names=tag_names,
        ),
    )

    return None
