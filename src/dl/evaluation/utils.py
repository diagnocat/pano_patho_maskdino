from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from detectron2.structures import BoxMode, Instances, polygons_to_bitmask

from ...etl.annotation import CONDITION_CLASS_TO_LABEL_PREDICTIONS
from .annotation import ImageAnnotation, InstanceAnnotation


def convert_dataset_dict_to_image_annotation(
    dataset_dict: dict[str, Any],
    tags_meta: dict[str, dict[int, str]] | None = None,
) -> ImageAnnotation:
    if tags_meta is None:
        tags_meta = {}

    instance_annotations = []
    for anno in dataset_dict["annotations"]:
        mask = polygons_to_bitmask(
            anno["segmentation"], dataset_dict["height"], dataset_dict["width"]
        )
        mask_rle = mask2rle(mask)
        bbox = BoxMode.convert(anno["bbox"], anno["bbox_mode"], BoxMode.XYXY_ABS)
        bbox = cast(tuple[float, float, float, float], tuple(bbox))
        instance_annotation = InstanceAnnotation(
            category_id=anno["category_id"],
            mask_rle=mask_rle,
            bbox=bbox,
            score=1.0,
            tags={tag_name: anno[tag_name] for tag_name in tags_meta},
        )
        instance_annotations.append(instance_annotation)

    image_annotation = ImageAnnotation(
        instance_annotations=instance_annotations,
        height=dataset_dict["height"],
        width=dataset_dict["width"],
        image_id=dataset_dict["image_id"],
    )
    return image_annotation


def convert_instances_to_image_annotation(
    instances: Instances,
    image_id: int,
    tags_meta: dict[str, dict[int, str]] | None = None,
) -> ImageAnnotation:
    if tags_meta is None:
        tags_meta = {}
    pred_classes = instances.pred_classes
    pred_boxes = instances.pred_boxes.tensor.long()
    assert pred_boxes.shape[-1] == 4
    pred_scores = instances.scores
    pred_masks = instances.pred_masks
    pred_tags = {}
    for tag_name in tags_meta:
        pred_tags[tag_name] = instances.get(f"{tag_name}_classes").numpy()

    instance_annotations = []
    for i in range(len(instances)):
        mask_rle = mask2rle(pred_masks[i].numpy())
        bbox = cast(tuple[float, float, float, float], tuple(pred_boxes[i].tolist()))
        instance_annotation = InstanceAnnotation(
            category_id=pred_classes[i].item(),
            mask_rle=mask_rle,
            bbox=bbox,
            score=pred_scores[i].item(),
            tags={tag_name: pred_tags[tag_name][i] for tag_name in tags_meta},
        )
        instance_annotations.append(instance_annotation)

    height, width = instances.image_size
    image_annotation = ImageAnnotation(
        instance_annotations=instance_annotations,
        height=height,
        width=width,
        image_id=image_id,
    )
    return image_annotation


def convert_pipelines_preds_to_image_annotation(
    pathologies: list[dict[str, Any]],
    image_id: int,
    height: int,
    width: int,
    code_to_resarch_condition: dict[str, str],
) -> ImageAnnotation:
    instance_annotations = []
    for patho in pathologies:
        if not patho["model_positive"]:
            continue

        if isinstance(patho["condition_code"], int):
            condition = code_to_resarch_condition[patho["condition_code"]]
        else:
            condition = patho["code"]

        if (category_id := CONDITION_CLASS_TO_LABEL_PREDICTIONS.get(condition)) is None:
            continue

        mask_rle = mask2rle(patho["mask"].astype(np.uint8))
        bbox = (
            patho["bbox"].xmin,
            patho["bbox"].ymin,
            patho["bbox"].xmax,
            patho["bbox"].ymax,
        )

        instance_annotation = InstanceAnnotation(
            category_id=category_id,
            mask_rle=mask_rle,
            bbox=bbox,
            score=1.0,
            tags={},
        )
        instance_annotations.append(instance_annotation)

    image_annotation = ImageAnnotation(
        instance_annotations=instance_annotations,
        height=height,
        width=width,
        image_id=image_id,
    )
    return image_annotation


def mask2rle(img: np.ndarray) -> str:
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
    figsize: tuple[int, int] = (20, 20),
    annotate: bool = False,
):
    """
    Plots a confusion matrix using matplotlib and seaborn.

    :param conf_matrix: numpy ndarray, confusion matrix of size N x N
    :param class_names: list, list of class names of length N
    """
    _, _ = plt.subplots(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=annotate,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
