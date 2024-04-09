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

    inference_driver = create_inference_driver(
        exp_path=exp_path,
        score_thresh=0.05,
        nms_thresh=nms_thresh,
        device=torch.device("cuda"),
        is_jit_scripted=is_jit_scripted,
    )

    cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
    tag_name_to_num_classes = cfg.INPUT.get("TAG_NAME_TO_NUM_CLASSES", {})
    tag_names = list(tag_name_to_num_classes)

    register_available_datasets(tag_names=tag_names)
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

    category_id_to_name_mapping = {
        i: category_name for i, category_name in enumerate(metadata.thing_classes)
    }

    results = cast(
        list[tuple[float, float]],
        Parallel(n_jobs=-1)(
            delayed(optimize_threshold_for_condition)(
                pred_image_annos=pred_image_annos,
                gt_image_annos=gt_image_annos,
                category_id=category_id,
                category_id_to_name_mapping=category_id_to_name_mapping,
                iou_thresh=iou_thresh,
                mask_iou=mask_iou,
            )
            for category_id in tqdm(
                category_id_to_name_mapping, desc="Optimizing thresholds"
            )
        ),
    )
    out = {}
    for category_id, category_name in category_id_to_name_mapping.items():
        threshold, f1_score = results[category_id]
        out[category_name] = threshold
        if verbose:
            loguru.logger.info(
                f"{category_name}: thresh: {threshold:.2f} F1: {f1_score:.3f}"
            )

    # dump to json
    with open(exp_path / "thresholds.json", "w") as f:
        json.dump(out, f, indent=4)


def optimize_threshold_for_condition(
    pred_image_annos: dict[int, ImageAnnotation],
    gt_image_annos: dict[int, ImageAnnotation],
    category_id: int,
    category_id_to_name_mapping: dict[int, str],
    iou_thresh: float,
    mask_iou: bool,
) -> tuple[float, float]:
    pred_image_annos = filter_annotations(pred_image_annos, category_id=category_id)
    gt_image_annos = filter_annotations(gt_image_annos, category_id=category_id)

    best_threshold = 1.0
    best_f1_score = 0.0
    for threshold in np.linspace(0.05, 0.95, 10):
        preds = filter_annotations(pred_image_annos, threshold=threshold)
        metrics = calculate_metrics(
            preds,
            gt_image_annos,
            category_id_to_name_mapping=category_id_to_name_mapping,
            tags_meta=None,
            verbose=False,
            iou_thresh=iou_thresh,
            mask_iou=mask_iou,
        )
        f1_score = metrics["per_class"][category_id_to_name_mapping[category_id]]["F1"]
        if f1_score > best_f1_score:
            best_threshold = threshold
            best_f1_score = f1_score

    return best_threshold, best_f1_score


def filter_annotations(
    image_annos: dict[int, ImageAnnotation],
    threshold: float | None = None,
    category_id: int | None = None,
) -> dict[int, ImageAnnotation]:
    out = {}
    for image_id, image_anno in image_annos.items():
        instance_annotations = image_anno["instance_annotations"]

        if threshold is not None:
            instance_annotations = [
                anno for anno in instance_annotations if anno["score"] > threshold
            ]
        if category_id is not None:
            instance_annotations = [
                anno
                for anno in instance_annotations
                if anno["category_id"] == category_id
            ]

        out[image_id] = ImageAnnotation(
            width=image_anno["width"],
            height=image_anno["height"],
            image_id=image_anno["image_id"],
            instance_annotations=instance_annotations,
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="val")
    args = parser.parse_args()
    main(
        args.exp_name,
        args.dataset,
    )
