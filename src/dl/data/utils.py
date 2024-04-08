from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, BoxMode, PolygonMasks, polygons_to_bitmask
from detectron2.utils.file_io import PathManager
from pycocotools import mask as coco_mask
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


def rle_to_poly(rle: dict) -> list:
    countouts, _ = cv2.findContours(
        maskUtils.decode(rle),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    main_countour = max(countouts, key=cv2.contourArea).flatten()
    segmentation = [main_countour]
    return segmentation


def dataset_dict_segm_to_poly(dataset_dict: dict[str, Any]) -> dict[str, Any]:
    for anno in dataset_dict["annotations"]:
        if isinstance(anno["segmentation"], dict):
            anno["segmentation"] = rle_to_poly(rle=anno["segmentation"])
    return dataset_dict


def load_coco_json(
    json_file: str,
    image_root: str,
    dataset_name: str,
    extra_annotation_keys: list[str] | None = None,
    tag_names: list[str] | None = None,
):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

    if extra_annotation_keys is None:
        extra_annotation_keys = []

    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    meta = MetadataCatalog.get(dataset_name)
    meta.json_file = json_file
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    tags = {}
    for cat in cats:
        if (cat_tags := cat.get("tags")) is not None:
            tags.update(cat_tags)

    if tag_names is None:
        # use all tags
        tag_names = list(tags)
    meta_tags = {}
    for tag_name in tag_names:
        if (tag_label_to_class := tags.get(tag_name)) is None:
            logger.warning(f"Tag {tag_name} is not found!")
        else:
            meta_tags[tag_name] = {
                int(label): class_name
                for label, class_name in tag_label_to_class.items()
            }
    meta.tags = meta_tags

    img_ids = sorted(coco_api.imgs)
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info(
        "Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file)
    )

    dataset_dicts = []

    ann_keys = ["bbox"] + extra_annotation_keys

    num_instances_without_valid_segmentation = 0

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            if "bbox" in anno and len(anno["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["category_id"] = id_map[anno["category_id"]]
            for tag_name in meta.tags:
                obj[tag_name] = anno.get(tag_name, -100)

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def convert_coco_poly_to_mask(
    segmentations: list[float], height: float, width: float
) -> torch.Tensor:
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def resolve_masks_from_annos(
    annos: list[dict], mask_format: str, image_size: tuple
) -> BitMasks | None:
    if len(annos) == 0 or "segmentation" not in annos[0]:
        return None

    segms = [obj["segmentation"] for obj in annos]
    if mask_format == "polygon":
        # TODO check type and provide better error
        masks = PolygonMasks(segms)
    else:
        assert mask_format == "bitmask", mask_format
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image_size))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert (
                    segm.ndim == 2
                ), "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a full-image segmentation mask "
                    "as a 2D ndarray.".format(type(segm))
                )
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )

    return masks
