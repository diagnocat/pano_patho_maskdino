import argparse
import json
from typing import Literal, cast

import cv2
import loguru
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from joblib import Parallel, delayed
from tqdm import tqdm

from src.defs import ROOT
from src.dl.data.register_coco import register_available_datasets
from src.dl.evaluation.annotation import ImageAnnotation
from src.dl.evaluation.metrics import calculate_metrics
from src.dl.evaluation.utils import (
    convert_dataset_dict_to_image_annotation,
    convert_instances_to_image_annotation,
)
from src.dl.inference.utils import create_inference_driver
from src.etl.annotation import (
    BINARY_TAG_TO_POSITIVE_CLASS,
    CONDITION_CLASS_TO_LABEL_PREDICTIONS,
    CONDITION_LABEL_TO_CLASS_PREDICTIONS,
    TAG_TO_CLASS_TO_LABEL,
    TAG_TO_CONDITIONS,
)
from src.maskdino.src.config.load import load_config


def main(
    exp_name: str,
    dataset: Literal["train", "val"] = "val",
    iou_thresh: float = 0.2,
    nms_thresh: float = 0.8,
    mask_iou: bool = False,
    is_jit_scripted: bool = False,
    verbose: bool = True,
) -> None:
    exp_path = ROOT / "outputs" / exp_name
    eval_dir = exp_path / f"eval-{dataset}"
    eval_dir.mkdir(exist_ok=True)

    inference_driver = create_inference_driver(
        exp_path=exp_path,
        score_thresh=0.15,
        nms_thresh=nms_thresh,
        device=torch.device("cuda"),
        is_jit_scripted=is_jit_scripted,
    )

    cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
    tag_name_to_num_classes = cfg.INPUT.get("TAG_NAME_TO_NUM_CLASSES", {})
    tag_names = list(tag_name_to_num_classes)

    register_available_datasets(tag_names=tag_names)
    # dataset_dicts = DatasetCatalog.get(f"coco-{dataset}") + DatasetCatalog.get(
    #     f"coco-test"
    # )
    dataset_dicts = DatasetCatalog.get(f"coco-{dataset}")
    metadata = MetadataCatalog.get(f"coco-{dataset}")

    gt_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts):
        image_annotation = convert_dataset_dict_to_image_annotation(
            dataset_dict, tags_meta=tag_name_to_num_classes
        )
        gt_image_annos[image_annotation["image_id"]] = image_annotation

    pred_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts, desc="Running inference"):
        image = cv2.imread(dataset_dict["file_name"])
        instances = inference_driver(
            image,
            rescale_scores_inplace=False,
            filter_by_threshold=True,
            resize_outputs_to_original_shape=True,
        )
        image_id = dataset_dict["image_id"]
        image_anno = convert_instances_to_image_annotation(
            instances, image_id, tag_name_to_num_classes
        )
        pred_image_annos[image_id] = image_anno

    results = cast(
        list[tuple[float, float]],
        Parallel(n_jobs=-1)(
            delayed(optimize_threshold_for_condition)(
                pred_image_annos=pred_image_annos,
                gt_image_annos=gt_image_annos,
                condition=condition,
                iou_thresh=iou_thresh,
                mask_iou=mask_iou,
            )
            for condition in tqdm(
                CONDITION_CLASS_TO_LABEL_PREDICTIONS,
                desc="Optimizing condition thresholds",
            )
        ),
    )

    condition_thresholds = {}
    for i, condition in enumerate(CONDITION_CLASS_TO_LABEL_PREDICTIONS):
        threshold, f1_score = results[i]
        condition_thresholds[condition] = threshold
        if verbose:
            loguru.logger.info(
                f"{condition}: thresh: {threshold:.2f} F1: {f1_score:.3f}"
            )

    results = cast(
        list[tuple[float, float]],
        Parallel(n_jobs=-1)(
            delayed(optimize_threshold_for_tag)(
                pred_image_annos=pred_image_annos,
                gt_image_annos=gt_image_annos,
                condition_thresholds=condition_thresholds,
                tags_meta=metadata.tags,
                tag_name=tag_name,
                positive_class=positive_class,
                iou_thresh=iou_thresh,
                mask_iou=mask_iou,
            )
            for tag_name, positive_class in tqdm(
                BINARY_TAG_TO_POSITIVE_CLASS.items(), desc="Optimizing tag thresholds"
            )
        ),
    )

    tag_thresholds = {}
    for i, tag_name in enumerate(BINARY_TAG_TO_POSITIVE_CLASS):
        threshold, f1_score = results[i]
        tag_thresholds[tag_name] = threshold
        if verbose:
            loguru.logger.info(
                f"{tag_name}: thresh: {threshold:.2f} F1: {f1_score:.3f}"
            )

    with open(exp_path / "condition_thresholds.json", "w") as f:
        json.dump(condition_thresholds, f, indent=4)

    with open(exp_path / "tag_thresholds.json", "w") as f:
        json.dump(tag_thresholds, f, indent=4)


def optimize_threshold_for_condition(
    pred_image_annos: dict[int, ImageAnnotation],
    gt_image_annos: dict[int, ImageAnnotation],
    condition: str,
    iou_thresh: float,
    mask_iou: bool,
) -> tuple[float, float]:
    condition_label = CONDITION_CLASS_TO_LABEL_PREDICTIONS[condition]
    gt_image_annos = filter_annotations(
        gt_image_annos, thresholds=None, condition_labels=[condition_label]
    )

    best_threshold = 1.0
    best_f1_score = 0.0
    for threshold in np.linspace(0.15, 0.85, 15):
        preds = filter_annotations(
            pred_image_annos, thresholds=[threshold], condition_labels=[condition_label]
        )
        metrics = calculate_metrics(
            preds,
            gt_image_annos,
            category_id_to_name_mapping=CONDITION_LABEL_TO_CLASS_PREDICTIONS,
            tags_meta=None,
            verbose=False,
            iou_thresh=iou_thresh,
            mask_iou=mask_iou,
        )
        f1_score = metrics["per_class"][condition]["F1"]
        if f1_score > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1_score

    return best_threshold, best_f1_score


def optimize_threshold_for_tag(
    pred_image_annos: dict[int, ImageAnnotation],
    gt_image_annos: dict[int, ImageAnnotation],
    condition_thresholds: dict[str, float],
    tags_meta: dict[str, dict[int, str]],
    tag_name: str,
    positive_class: str,
    iou_thresh: float,
    mask_iou: bool,
) -> tuple[float, float]:
    conditions = TAG_TO_CONDITIONS[tag_name]
    condition_labels = [
        CONDITION_CLASS_TO_LABEL_PREDICTIONS[condition] for condition in conditions
    ]
    thresholds = [condition_thresholds[condition] for condition in conditions]
    pred_image_annos = filter_annotations(
        pred_image_annos, thresholds=thresholds, condition_labels=condition_labels
    )
    gt_image_annos = filter_annotations(
        gt_image_annos, thresholds=None, condition_labels=condition_labels
    )

    best_threshold = 1.0
    best_f1_score = 0.0
    for threshold in np.linspace(0.15, 0.85, 15):
        preds = {
            image_id: resolve_tag_classes_by_threshold(anno, tag_name, threshold)
            for image_id, anno in pred_image_annos.items()
        }
        metrics = calculate_metrics(
            preds,
            gt_image_annos,
            category_id_to_name_mapping=CONDITION_LABEL_TO_CLASS_PREDICTIONS,
            tags_meta=tags_meta,
            verbose=False,
            iou_thresh=iou_thresh,
            mask_iou=mask_iou,
        )
        f1_score = metrics["per_tag"][tag_name][f"{positive_class}/f1_score"]
        if f1_score > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1_score

    return best_threshold, best_f1_score


def filter_annotations(
    image_annos: dict[int, ImageAnnotation],
    thresholds: list[float] | None,
    condition_labels: list[int],
) -> dict[int, ImageAnnotation]:
    out = {}
    if thresholds is None:
        thresholds = [0.0] * len(condition_labels)

    if len(thresholds) != len(condition_labels):
        raise ValueError(
            "Length of thresholds should be equal to length of condition_labels"
        )

    for image_id, image_anno in image_annos.items():
        instance_annotations = []

        for condition_label, threshold in zip(condition_labels, thresholds):
            instance_annotations.extend(
                [
                    anno
                    for anno in image_anno["instance_annotations"]
                    if anno["score"] >= threshold
                    and anno["category_id"] == condition_label
                ]
            )

        out[image_id] = ImageAnnotation(
            width=image_anno["width"],
            height=image_anno["height"],
            image_id=image_anno["image_id"],
            instance_annotations=instance_annotations,
        )
    return out


def resolve_tag_classes_by_threshold(
    pred_image_anno: ImageAnnotation, tag_name: str, threshold: float
):
    positive_class = BINARY_TAG_TO_POSITIVE_CLASS[tag_name]
    positive_label = TAG_TO_CLASS_TO_LABEL[tag_name][positive_class]
    assert positive_label in (0, 1)
    for instance_anno in pred_image_anno["instance_annotations"]:
        if tag_name not in instance_anno["tags"]:
            continue
        if "tags_full_scores" not in instance_anno:
            raise ValueError("tags_full_scores should not be present")

        if instance_anno["tags_full_scores"][tag_name][positive_label] > threshold:
            instance_anno["tags"][tag_name] = positive_label
        else:
            instance_anno["tags"][tag_name] = int(not positive_label)
    return pred_image_anno


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="val")
    args = parser.parse_args()
    main(
        args.exp_name,
        args.dataset,
    )
