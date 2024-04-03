from typing import Literal

import torch
import torch.nn
from detectron2.structures import Instances
from typing_extensions import assert_never

from ...etl.annotation import CONDITION_CLASS_TO_LABEL_PREDICTIONS


def apply_instances_nms(
    instances: Instances,
    nms_threshold: float,
    nms_threshold_class_agnostic: float = 0.98,
    nms_threshold_special_classes: float = 0.95,
) -> Instances:
    if len(instances) <= 2:
        return instances

    special_nms_classes = torch.tensor(
        [CONDITION_CLASS_TO_LABEL_PREDICTIONS[c] for c in ("canal_filling",)],
        dtype=torch.long,
        device=instances.scores.device,
    )
    special_nms_mask = torch.isin(instances.pred_classes, special_nms_classes)

    instances_special = instances[special_nms_mask]
    instances_common = instances[~special_nms_mask]

    indices_special = apply_mask_nms(
        instances_special.pred_masks,
        instances_special.scores,
        nms_threshold_special_classes,
        metric="iof",
    )
    instances_special = instances_special[indices_special]

    indices_common = apply_mask_nms(
        instances_common.pred_masks,
        instances_common.scores,
        nms_threshold,
        instances_common.pred_classes,
        metric="iof",
    )
    instances_common = instances_common[indices_common]

    indices_common_class_agnostic = apply_mask_nms(
        instances_common.pred_masks,
        instances_common.scores,
        nms_threshold_class_agnostic,
        metric="iou",
    )
    instances_common = instances_common[indices_common_class_agnostic]

    return Instances.cat([instances_special, instances_common])


def apply_mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    nms_threshold: float,
    pred_classes: torch.Tensor | None = None,
    metric: Literal["iou", "iof"] = "iof",
    eps: float = 1e-6,
) -> torch.Tensor:
    # Number of masks
    n_instances = masks.shape[0]

    if pred_classes is None:
        # If pred_classes is not provided, we perform class agnostic nms
        pred_classes = torch.ones(n_instances, dtype=torch.long, device=masks.device)

    # Tensor to keep track of which masks to keep
    keep = torch.zeros(n_instances, dtype=torch.bool, device=masks.device)

    # Sort scores in descending order
    _, sorted_indices = scores.sort(descending=True)

    for i in range(n_instances):
        i_mask = masks[sorted_indices[i]]

        # Check if this mask overlaps significantly with any previously selected mask
        overlap = False
        for j in range(i):
            if (
                keep[sorted_indices[j]]
                and pred_classes[sorted_indices[i]] == pred_classes[sorted_indices[j]]
            ):
                j_mask = masks[sorted_indices[j]]

                intersection = (i_mask & j_mask).sum().float()
                match metric:
                    case "iou":
                        union = (i_mask | j_mask).sum().float()
                    case "iof":
                        if i_mask.sum() <= j_mask.sum():
                            smallest_mask = i_mask
                        else:
                            smallest_mask = j_mask
                        union = smallest_mask.sum().float()
                    case _:
                        assert_never(metric)

                iou = intersection / (union + eps)

                if iou > nms_threshold:
                    overlap = True
                    break

        if not overlap:
            keep[sorted_indices[i]] = True

    return keep
