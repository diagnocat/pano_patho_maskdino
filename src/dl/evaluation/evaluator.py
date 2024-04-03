import itertools
import logging

import detectron2.utils.comm as comm
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.evaluation import DatasetEvaluator

from ...etl.annotation import CONDITION_CLASSES
from ..inference.postprocess import postprocess_instances
from .annotation import ImageAnnotation
from .metrics import calculate_metrics
from .utils import (
    convert_dataset_dict_to_image_annotation,
    convert_instances_to_image_annotation,
)


class CustomDatasetEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name: str,
        distributed: bool = True,
        score_thresh: float = 0.3,
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self.thresholds = {condition: score_thresh for condition in CONDITION_CLASSES}

        dataset_dicts = get_detection_dataset_dicts(dataset_name, filter_empty=False)
        self._metadata = MetadataCatalog.get(dataset_name)
        self.tags_meta = self._metadata.get("tags", {})

        self.gt_image_annos: dict[int, ImageAnnotation] = {}
        for dataset_dict in dataset_dicts:
            image_annotation = convert_dataset_dict_to_image_annotation(
                dataset_dict, self.tags_meta
            )
            self.gt_image_annos[image_annotation["image_id"]] = image_annotation

    def reset(self):
        self.pred_image_annos_list: list[ImageAnnotation] = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to(torch.device("cpu"))
            instances = postprocess_instances(
                instances=instances,
                probability_thresholds=self.thresholds,
            )
            image_annotation = convert_instances_to_image_annotation(
                instances, input["image_id"], self.tags_meta
            )
            self.pred_image_annos_list.append(image_annotation)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            pred_image_annos_list = comm.gather(self.pred_image_annos_list, dst=0)
            pred_image_annos_list = list(itertools.chain(*pred_image_annos_list))

            if not comm.is_main_process():
                return {}
        else:
            pred_image_annos_list = self.pred_image_annos_list

        pred_image_annos = {}
        for image_annotation in pred_image_annos_list:
            pred_image_annos[image_annotation["image_id"]] = image_annotation

        metrics = calculate_metrics(
            pred_image_annos,
            self.gt_image_annos,
            category_id_to_name_mapping={
                i: category_name
                for i, category_name in enumerate(self._metadata.thing_classes)
            },
            tags_meta=self.tags_meta,
            verbose=False,
            iou_thresh=0.2,
            mask_iou=True,
        )
        out = {}
        metrics_dict = {**metrics["per_class"], **metrics["per_tag"]}
        for name, metrics_ in metrics_dict.items():
            out.update(
                {
                    f"{metric_name}/{name}": metric_value
                    for metric_name, metric_value in metrics_.items()
                }
            )

        return out
