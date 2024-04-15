from __future__ import annotations

from typing import Dict, List, TypedDict

import loguru
import numpy as np
import pandas as pd
import scipy.optimize

from ...etl.annotation import BINARY_TAG_TO_POSITIVE_CLASS, TAG_TO_LABEL_TO_CLASS
from .annotation import ImageAnnotation, InstanceAnnotation
from .utils import rle2mask


class DetectionMetrics(TypedDict):
    TP: int
    FP: int
    FN: int


class Metrics(TypedDict):
    per_class: dict[str, dict[str, float]]
    per_tag: dict[str, dict[str, float]]
    conf_matrix_per_tag: dict[str, np.ndarray]


def calculate_metrics(
    pred_image_annos: dict[int, ImageAnnotation],
    gt_image_annos: dict[int, ImageAnnotation],
    category_id_to_name_mapping: dict[int, str],
    tags_meta: dict[str, dict[int, str]] | None = None,
    verbose: bool = True,
    iou_thresh: float = 0.0,
    eps: float = 1e-8,
    mask_iou: bool = True,
) -> Metrics:
    assert set(pred_image_annos) == set(gt_image_annos)
    if tags_meta is None:
        tags_meta = {}

    detection_metrics = {}
    segmentation_metrics = {
        class_name: [] for class_name in category_id_to_name_mapping.values()
    }
    conf_matrix_per_tag = {}

    for image_id, gt_image_anno in gt_image_annos.items():
        if (pred_image_anno := pred_image_annos.get(image_id)) is None:
            loguru.logger.warning(f"Image {image_id} is not present in predictions")
            continue

        assert pred_image_anno["height"] == gt_image_anno["height"]
        assert pred_image_anno["width"] == gt_image_anno["width"]

        resolve_instance_masks_inplace(gt_image_anno)
        resolve_instance_masks_inplace(pred_image_anno)

        detection_metrics_per_class = {}
        segmentation_metrics_per_class = {}

        for class_id, class_name in category_id_to_name_mapping.items():
            gt_annos_class = [
                anno
                for anno in gt_image_anno["instance_annotations"]
                if anno["category_id"] == class_id
            ]
            pred_annos_class = [
                anno
                for anno in pred_image_anno["instance_annotations"]
                if anno["category_id"] == class_id
            ]

            pred2gt_indices, gt2pred_indices = match_pred_and_gt_instances(
                gt_annos_class,
                pred_annos_class,
                iou_thresh=iou_thresh,
                mask_iou=mask_iou,
            )

            detection_metrics_class = calculate_detection_metrics_from_indices(
                pred2gt_indices=pred2gt_indices,
                gt2pred_indices=gt2pred_indices,
            )
            detection_metrics_per_class[class_name] = detection_metrics_class

            conf_matrix_per_tag_class = calculate_conf_matrix_per_tag(
                gt_annos=gt_annos_class,
                pred_annos=pred_annos_class,
                pred2gt_indices=pred2gt_indices,
                gt2pred_indices=gt2pred_indices,
                tags_meta=tags_meta,
            )
            for tag_name, conf_matrix in conf_matrix_per_tag_class.items():
                if tag_name in conf_matrix_per_tag:
                    conf_matrix_per_tag[tag_name] += conf_matrix
                else:
                    conf_matrix_per_tag[tag_name] = conf_matrix

            iou_class = calculate_iou_tp(
                gt_annos=gt_annos_class,
                pred_annos=pred_annos_class,
                pred2gt_indices=pred2gt_indices,
                image_height=gt_image_anno["height"],
                image_width=gt_image_anno["width"],
            )
            if iou_class is not None:
                segmentation_metrics_per_class[class_name] = iou_class

        detection_metrics = aggregate_detection_metrics(
            detection_metrics, detection_metrics_per_class
        )
        segmentation_metrics = aggregate_segmentation_metrics(
            segmentation_metrics, segmentation_metrics_per_class
        )

        remove_instance_masks_inplace(gt_image_anno)
        remove_instance_masks_inplace(pred_image_anno)

    assert len(segmentation_metrics) == len(category_id_to_name_mapping)
    assert len(detection_metrics) == len(category_id_to_name_mapping)

    metrics_per_class = {
        class_name: {} for class_name in category_id_to_name_mapping.values()
    }
    for class_name, ious in segmentation_metrics.items():
        miou = np.mean(ious) if len(ious) > 0 else 1.0
        metrics_per_class[class_name]["IoU"] = miou
        if verbose:
            loguru.logger.info(f"IoU for {class_name} = {np.mean(miou):.3f}")

    for class_name, detection_metrics_class in detection_metrics.items():
        recall = (detection_metrics_class["TP"] + eps) / (
            detection_metrics_class["TP"] + detection_metrics_class["FN"] + eps
        )
        precision = (detection_metrics_class["TP"] + eps) / (
            detection_metrics_class["TP"] + detection_metrics_class["FP"] + eps
        )
        f1_score = (2 * precision * recall) / (precision + recall + eps)

        if verbose:
            loguru.logger.info(f"Recall for {class_name} = {recall:.3f}")
            loguru.logger.info(f"Precision for {class_name} = {precision:.3f}")

        metrics_per_class[class_name]["Recall"] = recall
        metrics_per_class[class_name]["Precision"] = precision
        metrics_per_class[class_name]["F1"] = f1_score
        metrics_per_class[class_name]["TP"] = detection_metrics_class["TP"]
        metrics_per_class[class_name]["FP"] = detection_metrics_class["FP"]
        metrics_per_class[class_name]["FN"] = detection_metrics_class["FN"]

    metrics_per_class_df = pd.DataFrame(metrics_per_class).round(3)

    if verbose:
        for metric in ("Recall", "Precision", "F1", "IoU"):
            metric_value = metrics_per_class_df.loc[metric].mean()
            loguru.logger.info(f"{metric} mean {metric_value:.3f}")

    metrics_per_tag = {tag_name: {} for tag_name in tags_meta}
    for tag_name, conf_matrix in conf_matrix_per_tag.items():
        conf_matrix_matched = conf_matrix[:-1, :-1]
        positive_class = BINARY_TAG_TO_POSITIVE_CLASS.get(tag_name)
        for tag_label, tag_class in TAG_TO_LABEL_TO_CLASS[tag_name].items():
            # Don't calculate precision/recall for negative class
            # in case of binary classification
            if positive_class is not None and tag_class != positive_class:
                continue
            precision = (conf_matrix_matched[tag_label, tag_label] + eps) / (
                conf_matrix_matched[tag_label, :].sum() + eps
            )
            recall = (conf_matrix_matched[tag_label, tag_label] + eps) / (
                conf_matrix_matched[:, tag_label].sum() + eps
            )
            f1_score = (2 * precision * recall) / (precision + recall + eps)
            metrics_per_tag[tag_name][f"{tag_class}/precision"] = precision
            metrics_per_tag[tag_name][f"{tag_class}/recall"] = recall
            metrics_per_tag[tag_name][f"{tag_class}/f1_score"] = f1_score

    return Metrics(
        per_class=metrics_per_class,
        per_tag=metrics_per_tag,
        conf_matrix_per_tag=conf_matrix_per_tag,
    )


def resolve_instance_masks_inplace(image_annotation: ImageAnnotation) -> None:
    width = image_annotation["width"]
    height = image_annotation["height"]
    for annotation in image_annotation["instance_annotations"]:
        mask = rle2mask(annotation["mask_rle"], (width, height))
        annotation["mask"] = mask
    return None


def remove_instance_masks_inplace(image_annotation: ImageAnnotation) -> None:
    for annotation in image_annotation["instance_annotations"]:
        annotation.pop("mask")
    return None


def match_pred_and_gt_instances(
    gt_annos: list[InstanceAnnotation],
    pred_annos: list[InstanceAnnotation],
    iou_thresh: float = 0.0,
    mask_iou: bool = True,
    is_one_to_many: bool = False,
) -> tuple[dict[int, int | None], dict[int, int | None]]:
    pred2gt_indices: dict[int, int | None] = {
        pred_id: None for pred_id in range(len(pred_annos))
    }
    gt2pred_indices: dict[int, int | None] = {
        gt_id: None for gt_id in range(len(gt_annos))
    }

    if len(gt_annos) > 0 and len(pred_annos) > 0:
        iou_matrix = calc_iou_matrix(
            gt_annos=gt_annos, pred_annos=pred_annos, mask_iou=mask_iou
        )

        if is_one_to_many:
            gt_indices = iou_matrix.argmax(axis=0)
            pred_indices = np.arange(len(pred_annos))
            for gt_i, pred_i in zip(gt_indices, pred_indices):
                if iou_matrix[gt_i, pred_i] >= iou_thresh:
                    pred2gt_indices[pred_i] = gt_i
                    gt2pred_indices[gt_i] = pred_i
        else:
            gt_indices, pred_indices = scipy.optimize.linear_sum_assignment(
                1 - iou_matrix
            )

            for gt_i, pred_i in zip(gt_indices, pred_indices):
                if iou_matrix[gt_i, pred_i] >= iou_thresh:
                    pred2gt_indices[pred_i] = gt_i
                    gt2pred_indices[gt_i] = pred_i

    return pred2gt_indices, gt2pred_indices


def calculate_conf_matrix_per_tag(
    gt_annos: list[InstanceAnnotation],
    pred_annos: list[InstanceAnnotation],
    pred2gt_indices: dict[int, int | None],
    gt2pred_indices: dict[int, int | None],
    tags_meta: dict[str, dict[int, str]] | None = None,
    ignore_label: int = -100,
) -> dict[str, np.ndarray]:
    if tags_meta is None:
        tags_meta = {}

    confusion_matrices = {}
    for tag_name, tag_id_to_name_mapping in tags_meta.items():
        n_tag_classes = len(tag_id_to_name_mapping)
        # last row and column are for unmatched instances
        confusion_matrix = np.zeros(
            (n_tag_classes + 1, n_tag_classes + 1), dtype=np.int64
        )
        for pred_i, gt_i in pred2gt_indices.items():
            pred_tag = pred_annos[pred_i]["tags"].get(tag_name)
            if gt_i is not None:
                gt_tag = gt_annos[gt_i]["tags"][tag_name]
            else:
                gt_tag = n_tag_classes
            if pred_tag != ignore_label and gt_tag != ignore_label:
                confusion_matrix[pred_tag, gt_tag] += 1

        for gt_i, pred_i in gt2pred_indices.items():
            if pred_i is None:
                pred_tag = n_tag_classes
                gt_tag = gt_annos[gt_i]["tags"][tag_name]
                if gt_tag != ignore_label:
                    confusion_matrix[pred_tag, gt_tag] += 1

        confusion_matrices[tag_name] = confusion_matrix

    return confusion_matrices


def calc_iou_matrix(
    gt_annos: list[InstanceAnnotation],
    pred_annos: list[InstanceAnnotation],
    mask_iou: bool = True,
) -> np.ndarray:
    """
    calculates iou matrix (len(gt), len(pred)) for gt vs predictions masks
    """
    iou_matrix = np.zeros((len(gt_annos), len(pred_annos)))
    for gt_i, gt_anno in enumerate(gt_annos):
        for pred_i, pred_anno in enumerate(pred_annos):
            if mask_iou:
                iou = calc_mask_iou(gt_anno["mask"], pred_anno["mask"])
            else:
                iou = calc_box_iou(gt_anno["bbox"], pred_anno["bbox"])
            iou_matrix[gt_i][pred_i] = iou
    return iou_matrix


def calc_mask_iou(
    mask_gt: np.ndarray, mask_pred: np.ndarray, eps: float = 1e-8
) -> float:
    intersection = (mask_gt * mask_pred).sum()
    union = mask_gt.sum() + mask_pred.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou


def calc_box_iou(
    box_gt: tuple[float, float, float, float],
    box_pred: tuple[float, float, float, float],
    eps: float = 1e-8,
) -> float:
    xg1, yg1, xg2, yg2 = box_gt
    xp1, yp1, xp2, yp2 = box_pred

    area_gt = (xg2 - xg1) * (yg2 - yg1)
    area_pred = (xp2 - xp1) * (yp2 - yp1)

    w = max(min(xp2, xg2) - max(xp1, xg1), 0)
    h = max(min(yp2, yg2) - max(yp1, yg1), 0)
    intersection = h * w
    iou = (intersection + eps) / (area_pred + area_gt - intersection + eps)

    return iou


def calculate_detection_metrics_from_indices(
    pred2gt_indices: dict[int, int | None],
    gt2pred_indices: dict[int, int | None],
) -> DetectionMetrics:
    metrics = {"TP": 0, "FN": 0, "FP": 0}

    for pred_i in gt2pred_indices.values():
        if pred_i is None:
            metrics["FN"] += 1
        else:
            metrics["TP"] += 1

    for gt_i in pred2gt_indices.values():
        if gt_i is None:
            metrics["FP"] += 1

    return DetectionMetrics(**metrics)


def calculate_iou_tp(
    gt_annos: list[InstanceAnnotation],
    pred_annos: list[InstanceAnnotation],
    pred2gt_indices: dict[int, int | None],
    image_height: int,
    image_width: int,
) -> float | None:
    if len(gt_annos) == 0 and len(pred_annos) == 0:
        return None

    masks_gt = []
    masks_pred = []
    for pred_i, gt_i in pred2gt_indices.items():
        if gt_i is None:
            continue
        gt_anno = gt_annos[gt_i]
        pred_anno = pred_annos[pred_i]
        if "mask" not in gt_anno or "mask" not in pred_anno:
            raise ValueError("Masks are not present in annotations")
        masks_gt.append(gt_anno["mask"])
        masks_pred.append(pred_anno["mask"])

    if len(masks_gt) > 0:
        masks_gt = np.stack(masks_gt, axis=0)
        mask_gt = masks_gt.any(axis=0).astype(np.uint8)

        masks_pred = np.stack(masks_pred, axis=0)
        mask_pred = masks_pred.any(axis=0).astype(np.uint8)
    else:
        mask_gt = np.zeros((image_height, image_width), dtype=np.uint8)
        mask_pred = np.zeros((image_height, image_width), dtype=np.uint8)

    iou = calc_mask_iou(mask_gt, mask_pred)
    if np.isnan(iou):
        loguru.logger.warning("NaN value in IoU calculation encountered!")

    return iou


def aggregate_detection_metrics(
    detection_metrics: Dict[str, DetectionMetrics],
    detection_metrics_case: Dict[str, DetectionMetrics],
) -> Dict[str, DetectionMetrics]:
    if len(detection_metrics) == 0:
        return detection_metrics_case
    for class_name, class_metrics in detection_metrics.items():
        for key in class_metrics:
            class_metrics[key] += detection_metrics_case[class_name][key]

    return detection_metrics


def aggregate_segmentation_metrics(
    segmentation_metrics: Dict[str, List[float]],
    segmentation_metrics_case: Dict[str, float],
) -> Dict[str, List[float]]:
    for class_name, iou in segmentation_metrics_case.items():
        segmentation_metrics[class_name].append(iou)
    return segmentation_metrics
