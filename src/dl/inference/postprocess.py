import numpy as np
import torch
from detectron2.structures import Instances

from ...etl.annotation import CONDITION_CLASS_TO_LABEL_PREDICTIONS, SURFACES


def postprocess_instances(
    instances: Instances,
    probability_thresholds: dict[str, float],
    rescale_scores_inplace: bool = True,
    filter_by_threshold: bool = True,
) -> Instances:
    instances.pred_boxes.clip(instances.image_size)

    instances_filtered = []
    for condition, threshold in probability_thresholds.items():
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

    instances = postprocess_tags(instances)

    instances = ensure_mask_within_bbox(instances)

    return instances


def postprocess_tags(instances: Instances, ignore_label: int = -100) -> Instances:
    tag_name_to_conditions = {
        "is_buildup": ["filling"],
        "post_material": ["post"],
        "involvement": ["caries", "secondary_caries", "filling"],
        **{
            f"is_surface_{surface}": ["caries", "secondary_caries", "filling"]
            for surface in SURFACES
        },
    }

    for tag_name, conditions in tag_name_to_conditions.items():
        if not instances.has(f"{tag_name}_classes"):
            continue

        conditions_mask = np.isin(
            instances.pred_classes.cpu().numpy(),
            np.array([CONDITION_CLASS_TO_LABEL_PREDICTIONS[c] for c in conditions]),
        )
        getattr(instances, f"{tag_name}_classes")[~conditions_mask] = ignore_label

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
