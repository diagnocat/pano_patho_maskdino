import copy
import logging
import random
from typing import Any

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, PolygonMasks

from .utils import convert_coco_poly_to_mask, resolve_masks_from_annos


class CustomDatasetMapper:
    def __init__(
        self,
        is_train: bool,
        transforms: list[T.Augmentation | T.Transform],
        image_format: str,
        tag_names: list[str] | None = None,
        dataset_dicts: list[dict[str, Any]] | None = None,
        mixup_proba: float = 0.0,
    ):
        self.transforms = transforms
        self.img_format = image_format
        self.is_train = is_train

        if tag_names is None:
            tag_names = []
        self.tag_names = tag_names

        if mixup_proba > 0.0 and dataset_dicts is None:
            raise ValueError("If mixup is enabled, dataset_dicts must be not None")

        self.mixup_proba = mixup_proba
        self.dataset_dicts = dataset_dicts

        logger = logging.getLogger("detectron2")
        logger.info(
            f"[CustomDatasetMapper] [{is_train=}] Transforms: {self.transforms}"
        )
        logger.info(f"[CustomDatasetMapper] [{is_train=}] Tag Names: {self.tag_names}")

    def __call__(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        if (random.uniform(0, 1) < self.mixup_proba) and self.is_train:
            dataset_dict = self.apply_mixup_transform(dataset_dict=dataset_dict)

        dataset_dict = self.apply_transforms(
            dataset_dict=dataset_dict, transforms=self.transforms
        )

        dataset_dict = self.convert_to_training_format(dataset_dict=dataset_dict)

        return dataset_dict

    def apply_transforms(
        self,
        dataset_dict: dict[str, Any],
        transforms: list[T.Augmentation | T.Transform] | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dataset_dict: a dataset dict after applied transforms to image and annotations
        """
        if (image := dataset_dict.get("image")) is None:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if transforms is None:
            transforms = self.transforms

        # it will be modified by code below
        dataset_dict = {
            k: copy.deepcopy(v) for k, v in dataset_dict.items() if k != "image"
        }

        image, transforms = T.apply_augmentations(transforms, image)
        dataset_dict["image"] = image

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        image_shape = image.shape[:2]  # h, w
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
        ]

        dataset_dict["annotations"] = annos
        return dataset_dict

    def convert_to_training_format(
        self, dataset_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        assert "image" in dataset_dict

        image = dataset_dict["image"]
        image_shape = image.shape[:2]
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if (annotations := dataset_dict.pop("annotations", None)) is None:
            return dataset_dict

        instances = annotations_to_instances(annotations, image_shape, self.tag_names)

        if not instances.has("gt_masks"):
            instances.gt_masks = PolygonMasks([])
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        # Generate masks from polygon
        h, w = instances.image_size
        if hasattr(instances, "gt_masks"):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks

        dataset_dict["instances"] = instances

        return dataset_dict

    def apply_mixup_transform(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        other_dataset_dict = random.choice(self.dataset_dicts)

        h1, w1 = dataset_dict["height"], dataset_dict["width"]
        h2, w2 = other_dataset_dict["height"], other_dataset_dict["width"]

        h, w = max(h1, h2), max(w1, w2)

        dataset_dict = self.apply_transforms(
            dataset_dict=dataset_dict,
            transforms=[T.PadTransform(x0=0, y0=0, x1=w - w1, y1=h - h1)],
        )
        other_dataset_dict = self.apply_transforms(
            dataset_dict=other_dataset_dict,
            transforms=[T.PadTransform(x0=0, y0=0, x1=w - w2, y1=h - h2)],
        )

        dataset_dict["annotations"] += other_dataset_dict["annotations"]

        blend_weight = np.random.beta(32.0, 32.0)
        bl_tfm = T.BlendTransform(
            src_image=other_dataset_dict["image"],
            src_weight=blend_weight,
            dst_weight=1 - blend_weight,
        )
        dataset_dict["image"] = bl_tfm.apply_image(dataset_dict["image"])
        return dataset_dict


def annotations_to_instances(
    annos: list[dict[str, Any]],
    image_size: tuple[int, int],
    tag_names: list[str],
    mask_format: str = "polygon",
) -> Instances:
    instances = Instances(image_size)

    boxes = [
        BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
        for obj in annos
    ]
    boxes = torch.as_tensor(
        np.array(boxes), dtype=torch.float32, device=torch.device("cpu")
    )
    instances.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    instances.gt_classes = classes

    for tag_name in tag_names:
        tag_values = [obj.get(tag_name, -100) for obj in annos]
        tag_values = torch.tensor(tag_values, dtype=torch.int64)
        instances.set(f"gt_{tag_name}", tag_values)

    masks = resolve_masks_from_annos(
        annos=annos, mask_format=mask_format, image_size=image_size
    )
    if masks is not None:
        instances.gt_masks = masks

        is_mask_annotated = [int(obj.get("is_mask_annotated", True)) for obj in annos]
        is_mask_annotated = torch.tensor(is_mask_annotated, dtype=torch.int64)
        instances.gt_is_mask_annotated = is_mask_annotated

    return instances
