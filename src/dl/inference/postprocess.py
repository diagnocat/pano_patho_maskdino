import numpy as np
import torch
from detectron2.structures import Instances

from ...etl.annotation import (
    BINARY_TAG_TO_POSITIVE_CLASS,
    CONDITION_CLASS_TO_LABEL_PREDICTIONS,
    TAG_TO_CLASS_TO_LABEL,
    TAG_TO_CONDITIONS,
)


def postprocess_instances(
    instances: Instances,
    condition_thresholds: dict[str, float],
    tag_thresholds: dict[str, float] | None = None,
    rescale_scores_inplace: bool = True,
    filter_by_threshold: bool = True,
) -> Instances:
    instances.pred_boxes.clip(instances.image_size)

    instances_filtered = []
    for condition, threshold in condition_thresholds.items():
        condition_mask: torch.Tensor = (
            instances.pred_classes == CONDITION_CLASS_TO_LABEL_PREDICTIONS[condition]
        )
        instances_condition = instances[condition_mask]
        if rescale_scores_inplace:
            instances_condition.scores = rescale_probabilities(
                instances_condition.scores, threshold
            )
            # Once rescaled, the optimal threshold is 0.5
            threshold = 0.5

        if filter_by_threshold:
            score_mask: torch.Tensor = instances_condition.scores > threshold
            instances_condition = instances_condition[score_mask]

        instances_filtered.append(instances_condition)

    instances = Instances.cat(instances_filtered)

    instances = postprocess_tags(instances, tag_thresholds)

    instances = ensure_mask_within_bbox(instances)

    return instances


def postprocess_tags(
    instances: Instances,
    tag_thresholds: dict[str, float] | None = None,
    ignore_label: int = -100,
) -> Instances:
    if tag_thresholds is None:
        tag_thresholds = {}

    for tag_name, conditions in TAG_TO_CONDITIONS.items():
        if not instances.has(f"{tag_name}_classes"):
            continue

        if (threshold := tag_thresholds.get(tag_name)) is not None:
            positive_class = BINARY_TAG_TO_POSITIVE_CLASS[tag_name]
            positive_label = TAG_TO_CLASS_TO_LABEL[tag_name][positive_class]
            negative_label = int(not positive_label)

            full_scores = instances.get(f"{tag_name}_full_scores")
            positive_scores = full_scores[:, positive_label]
            positive_scores = rescale_probabilities(positive_scores, threshold)
            is_positive_prediction = positive_scores >= 0.5

            scores = torch.where(
                is_positive_prediction,
                positive_scores,
                1 - positive_scores,
            )
            full_scores[:, positive_label] = positive_scores
            full_scores[:, negative_label] = 1 - positive_scores

            pred_classes = torch.where(
                is_positive_prediction, positive_label, negative_label
            )
            instances.set(f"{tag_name}_classes", pred_classes)
            instances.set(f"{tag_name}_scores", scores)
            instances.set(f"{tag_name}_full_scores", full_scores)

        conditions_mask = np.isin(
            instances.pred_classes.cpu().numpy(),
            np.array([CONDITION_CLASS_TO_LABEL_PREDICTIONS[c] for c in conditions]),
        )
        instances.get(f"{tag_name}_classes")[~conditions_mask] = ignore_label

    return instances


def ensure_mask_within_bbox(instances: Instances) -> Instances:
    boxes = instances.pred_boxes.tensor.long().cpu()
    masks = instances.pred_masks
    for i in range(len(instances)):
        x1, y1, x2, y2 = boxes[i].tolist()
        masks[i, :y1, :] = False
        masks[i, y2:, :] = False
        masks[i, :, :x1] = False
        masks[i, :, x2:] = False

    return instances


def rescale_probabilities(probas: torch.Tensor, threshold: float) -> torch.Tensor:
    """Rescale probabilities that 0.5 corresponds to the threshold.

    Args:
        probas (torch.Tensor): probabilities
        threshold (float): optimal threshold

    Returns:
        torch.Tensor: rescaled probabilities
    """
    return (
        probas * (1 - threshold) / (probas * (1 - threshold) + (1 - probas) * threshold)
    )
