import detectron2.data.transforms as T
import loguru
import numpy as np
import torch
import torch.nn as nn
from detectron2.structures import Boxes, Instances

from .nms import apply_instances_nms
from .postprocess import postprocess_instances


class IMaskdinoModule(nn.Module):
    tag_names: list[str]
    min_size_test: int
    max_size_test: int
    thresholds: dict[str, float]
    nms_thresh: float | None = None

    def forward(
        self, batched_inputs: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]: ...


class InferenceDriver:
    def __init__(
        self,
        model: IMaskdinoModule,
        device: torch.device,
        probability_thresholds: dict[str, float],
        nms_threshold: float | None = None,
        is_jit_scripted: bool = False,
        verbose: bool = True,
    ):
        self.model = model
        self.probability_thresholds = probability_thresholds
        self.is_jit_scripted = is_jit_scripted
        self.device = device
        self.nms_threshold = nms_threshold

        self.resize_transform = T.ResizeShortestEdge(
            short_edge_length=model.min_size_test,
            max_size=model.max_size_test,
            sample_style="choice",
        )

        if verbose:
            loguru.logger.info(f"Resizing transform: {self.resize_transform}")

    @torch.no_grad()
    def __call__(
        self,
        original_image: np.ndarray,
        rescale_scores_inplace: bool = True,
        filter_by_threshold: bool = True,
        crop: tuple[int, int, int, int] | None = None,
        resize_outputs_to_original_shape: bool = True,
    ) -> Instances:
        assert (
            original_image.ndim == 3 and original_image.shape[2] == 3
        ), original_image.shape

        if crop is not None:
            ymin_wf, xmin_wf, ymax_wf, xmax_wf = crop
            working_image = original_image[ymin_wf:ymax_wf, xmin_wf:xmax_wf]
        else:
            working_image = original_image

        resized_image = self.resize_transform.get_transform(working_image).apply_image(
            working_image
        )
        if resize_outputs_to_original_shape:
            model_output_height, model_output_width = working_image.shape[:2]
        else:
            model_output_height, model_output_width = resized_image.shape[:2]

        input_tensor = torch.as_tensor(
            resized_image.astype("float32").transpose(2, 0, 1), device=self.device
        )

        inputs = {
            "image": input_tensor,
            "height": torch.as_tensor(model_output_height),
            "width": torch.as_tensor(model_output_width),
        }

        if self.is_jit_scripted:
            with torch.jit.optimized_execution(False):
                instances_fields = self.model([inputs])[0]
        else:
            instances_fields = self.model([inputs])[0]

        instances = convert_fields_to_instances(
            instances_fields, model_output_height, model_output_width
        )

        instances = postprocess_instances(
            instances=instances,
            probability_thresholds=self.probability_thresholds,
            rescale_scores_inplace=rescale_scores_inplace,
            filter_by_threshold=filter_by_threshold,
        )
        if self.nms_threshold is not None:
            instances = apply_instances_nms(instances, self.nms_threshold)

        if crop is not None:
            original_height, original_width = original_image.shape[:2]
            if resize_outputs_to_original_shape:
                output_height, output_width = original_height, original_width
            else:
                working_field_height, working_field_width = working_image.shape[:2]
                resized_height, resized_width = resized_image.shape[:2]
                scale_height = resized_height / working_field_height
                scale_width = resized_width / working_field_width

                ymin_wf = int(ymin_wf * scale_height)
                ymax_wf = int(ymax_wf * scale_height)
                xmin_wf = int(xmin_wf * scale_width)
                xmax_wf = int(xmax_wf * scale_width)

                output_height = int(original_height * scale_height)
                output_width = int(original_width * scale_width)

            boxes = uncrop_boxes(instances.pred_boxes.tensor, xmin_wf, ymin_wf)
            masks = uncrop_masks(
                instances.pred_masks,
                xmin_wf,
                ymin_wf,
                xmax_wf,
                ymax_wf,
                (output_height, output_width),
            )
            instances = Instances(
                image_size=(output_height, output_width),
                **{
                    k: v
                    for k, v in instances.get_fields().items()
                    if k not in ("pred_boxes", "pred_masks")
                },
            )
            instances.pred_boxes = Boxes(boxes)
            instances.pred_masks = masks

        instances = instances.to("cpu")

        return instances


def convert_fields_to_instances(
    fields: dict[str, torch.Tensor], height: int, width: int
) -> Instances:
    instances = Instances((height, width))
    for k, v in fields.items():
        if k == "pred_boxes":
            v = Boxes(v)
        instances.set(k, v)
    return instances


def uncrop_masks(
    masks: torch.Tensor,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    shape: tuple[int, int],
) -> torch.Tensor:
    out = torch.zeros(
        (masks.shape[0], *shape),
        dtype=masks.dtype,
        device=masks.device,
    )
    out[:, ymin:ymax, xmin:xmax] = masks
    return out


def uncrop_boxes(
    boxes: torch.Tensor,
    xmin: int,
    ymin: int,
    inplace: bool = False,
) -> torch.Tensor:
    if inplace:
        out = boxes
    else:
        out = boxes.clone()
    out[:, 0::2] += xmin
    out[:, 1::2] += ymin
    return out
