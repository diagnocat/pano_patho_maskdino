from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from detectron2.data.catalog import Metadata
from detectron2.structures import Instances
from detectron2.utils.visualizer import _create_text_labels
from torch.utils.data import Dataset

from .visualizer import CustomVisualizer


def visualize_dataset_dict(
    d: Dict[str, Any],
    metadata: Metadata,
    scale: float = 3.0,
    figsize=(40, 40),
    tags_to_viz: list[str] | None = None,
    labels_to_viz: list[str] | None = None,
) -> np.ndarray:
    d = copy.deepcopy(d)

    if "image" in d:
        img = d["image"]
    else:
        img = cv2.imread(d["file_name"])[:, :, ::-1]
    visualizer = CustomVisualizer(img, metadata=metadata, scale=scale)
    out = visualizer.draw_dataset_dict(
        d, tags_to_viz=tags_to_viz, labels_to_viz=labels_to_viz
    )
    out = out.get_image()

    _, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].imshow(img)
    axarr[1].imshow(out)
    for ax in axarr:
        ax.axis("off")
    plt.show()
    return out


def visualize_batch_item(
    batch_item: Dict[str, Any], metadata: Metadata, figsize=(30, 20)
):
    image = batch_item["image"].permute(1, 2, 0).numpy()
    visualizer = CustomVisualizer(image, metadata=metadata, scale=1)
    if "instances" in batch_item:
        instances = batch_item["instances"].to("cpu")
        boxes = instances.gt_boxes
        classes = instances.gt_classes.tolist()
        labels = _create_text_labels(
            classes=classes,
            scores=None,
            class_names=metadata.get("thing_classes", None),
        )
        masks = instances.gt_masks

        out = visualizer.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
        )
        out = out.get_image()
    else:
        out = image

    _, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].imshow(image)
    axarr[1].imshow(out)
    for ax in axarr:
        ax.axis("off")
    plt.show()
    return out


def visualize_instance_predictions(
    image: np.ndarray,
    instances: Instances,
    metadata: Metadata,
    figsize: tuple[int, int] = (30, 20),
    scale: float = 3.0,
    tags_to_viz: list[str] | None = None,
    labels_to_viz: list[str] | None = None,
):
    visualizer = CustomVisualizer(image, metadata=metadata, scale=scale)
    out = visualizer.draw_instance_predictions(
        instances.to("cpu"), tags_to_viz=tags_to_viz, labels_to_viz=labels_to_viz
    )
    out = out.get_image()

    _, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].imshow(image)
    axarr[1].imshow(out)
    for ax in axarr:
        ax.axis("off")
    plt.show()
    return out


def visualize_comparison_dataset_dict_instances(
    d: Dict[str, Any],
    instances: Instances,
    metadata: Metadata,
    scale: float = 3.0,
    figsize=(40, 40),
    tags_to_viz: list[str] | None = None,
    labels_to_viz: list[str] | None = None,
):
    d = copy.deepcopy(d)

    if "image" in d:
        img = d["image"]
    else:
        img = cv2.imread(d["file_name"])[:, :, ::-1]
    visualizer = CustomVisualizer(img.copy(), metadata=metadata, scale=scale)
    dataset_dict_vis = visualizer.draw_dataset_dict(
        d, tags_to_viz=tags_to_viz, labels_to_viz=labels_to_viz
    )
    dataset_dict_vis = dataset_dict_vis.get_image()

    visualizer = CustomVisualizer(img.copy(), metadata=metadata, scale=scale)
    instances_vis = visualizer.draw_instance_predictions(
        instances.to("cpu"), tags_to_viz=tags_to_viz, labels_to_viz=labels_to_viz
    )
    instances_vis = instances_vis.get_image()

    _, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].imshow(dataset_dict_vis)
    axarr[0].set_title("GT")
    axarr[1].imshow(instances_vis)
    axarr[1].set_title("Prediction")
    for ax in axarr:
        ax.axis("off")
    plt.show()
    return None


def visualize_comparison_instances(
    image: np.ndarray,
    instances: Instances,
    other: Instances,
    metadata: Metadata,
    scale: float = 3.0,
    figsize=(40, 40),
    tags_to_viz: list[str] | None = None,
):
    visualizer = CustomVisualizer(image.copy(), metadata=metadata, scale=scale)
    instances_vis = visualizer.draw_instance_predictions(
        instances.to("cpu"), tags_to_viz=tags_to_viz
    )
    instances_vis = instances_vis.get_image()

    visualizer = CustomVisualizer(image.copy(), metadata=metadata, scale=scale)
    other_vis = visualizer.draw_instance_predictions(
        other.to("cpu"), tags_to_viz=tags_to_viz
    )
    other_vis = other_vis.get_image()

    _, axarr = plt.subplots(1, 2, figsize=figsize)
    axarr[0].imshow(instances_vis)
    axarr[1].imshow(other_vis)
    for ax in axarr:
        ax.axis("off")
    plt.show()
    return None


def plot_and_save_from_dataset(
    dataset: Dataset, metadata: Metadata, save_dir: str, n_save_pics: int
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    indices = np.random.randint(len(dataset), size=n_save_pics)
    for idx in indices:
        item = dataset[idx]
        fname = Path(item["file_name"])
        img = cv2.imread(str(fname))[:, :, ::1]
        PIL.Image.fromarray(img).save(
            Path(save_dir) / f"base_{fname.parent.stem}_{fname.stem}.png"
        )
        img_viz = visualize_batch_item(item, metadata, figsize=None)
        PIL.Image.fromarray(img_viz).save(
            Path(save_dir) / f"augs_{fname.parent.stem}_{fname.stem}.png"
        )
