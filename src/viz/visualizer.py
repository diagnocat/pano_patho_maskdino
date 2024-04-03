from __future__ import annotations

from typing import Any

import numpy as np
import torch
from detectron2.structures import BoxMode, Instances
from detectron2.utils.visualizer import GenericMask, Visualizer

# fmt: off
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)


class CustomVisualizer(Visualizer):
    def draw_dataset_dict(self, dic, tags_to_viz: list[str] | None = None, labels_to_viz: list[str] | None = None):
        """
        Draw annotations/segmentations in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic["annotations"]
        if labels_to_viz is not None:
            annos = [anno for anno in annos if self.metadata.thing_classes[anno["category_id"]] in labels_to_viz]
        if len(annos) == 0:
            return self.output

        if "segmentation" in annos[0]:
            masks = [x["segmentation"] for x in annos]
        else:
            masks = None
        if "keypoints" in annos[0]:
            keypts = [x["keypoints"] for x in annos]
            keypts = np.array(keypts).reshape(len(annos), -1, 3)
        else:
            keypts = None

        boxes = [
            BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos
        ]

        colors = [_COLORS[anno["category_id"]] for anno in annos]

        labels = self.create_text_labels_from_annos(annos, tags_to_viz=tags_to_viz)
        self.overlay_instances(
            labels=labels,
            boxes=boxes,
            masks=masks,
            keypoints=keypts,
            assigned_colors=colors,
        )

        return self.output

    def draw_instance_predictions(
        self, predictions: Instances, tags_to_viz: list[str] | None = None, labels_to_viz: list[str] | None = None,
    ):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """

        if labels_to_viz is not None:
            mask = torch.tensor([self.metadata.thing_classes[c] in labels_to_viz for c in predictions.pred_classes.tolist()])
            predictions = predictions[mask]

        if len(predictions) == 0:
            return self.output

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        labels = self.create_text_labels_from_instances(
            predictions, tags_to_viz=tags_to_viz
        )

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [
                GenericMask(x, self.output.height, self.output.width) for x in masks
            ]
        else:
            masks = None

        colors = [_COLORS[label] for label in predictions.pred_classes.tolist()]
        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            assigned_colors=colors
        )
        return self.output

    def create_text_labels_from_annos(
        self,
        annotations: list[dict[str, Any]],
        ignore_label: int = -100,
        tags_to_viz: list[str] | None = None,
    ) -> list[str]:
        labels = []
        metadata_tags = self.metadata.get("tags", {})
        if tags_to_viz is None:
            tags_to_viz = list(metadata_tags)
        for anno in annotations:
            label = self.metadata.thing_classes[anno["category_id"]]

            for tag_name, tag_label_to_class in metadata_tags.items():
                if tag_name not in tags_to_viz:
                    continue
                tag_label = anno[tag_name]
                if tag_label != ignore_label:
                    tag_class = tag_label_to_class[tag_label]
                    if not tag_class.startswith("Not "):
                        label += f" {tag_class}"
            labels.append(label)
        return labels

    def create_text_labels_from_instances(
        self,
        instances: Instances,
        ignore_label: int = -100,
        tags_to_viz: list[str] | None = None,
    ) -> list[str]:
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        tags_predictions = {}
        metadata_tags = self.metadata.get("tags", {})
        if tags_to_viz is None:
            tags_to_viz = list(metadata_tags)
        for tag_name in metadata_tags:
            if tag_name not in tags_to_viz:
                continue
            if instances.has(f"{tag_name}_scores"):
                tags_predictions[tag_name] = {
                    "scores": instances.get(f"{tag_name}_scores").tolist(),
                    "classes": instances.get(f"{tag_name}_classes").tolist(),
                }

        labels = []
        for i in range(len(instances)):
            label = self.metadata.thing_classes[classes[i]]
            label += f" {scores[i]:.2f}"
            for tag_name, tag_predictions in tags_predictions.items():
                tag_label = tag_predictions["classes"][i]
                if tag_label == ignore_label:
                    continue
                tag_score = tag_predictions["scores"][i]
                tag_class = metadata_tags[tag_name][tag_label]
                if not tag_class.startswith("Not "):
                    label += f" {tag_class} {tag_score:.2f}"
            labels.append(label)

        return labels
