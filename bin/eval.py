import argparse
from typing import Literal

import cv2
import loguru
import pandas as pd
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

from src.defs import ROOT
from src.dl.data.register_coco import register_available_datasets
from src.dl.evaluation.metrics import calculate_metrics
from src.dl.evaluation.utils import (
    convert_dataset_dict_to_image_annotation,
    convert_instances_to_image_annotation,
    plot_confusion_matrix,
)
from src.dl.inference.utils import create_inference_driver
from src.maskdino.src.config.load import load_config


def main(
    exp_name: str,
    score_thresh: float | None = None,
    dataset: Literal[
        "train", "val", "test", "test-orig", "test-pipelines", "test-pipelines-orig"
    ] = "test",
    is_jit_scripted: bool = False,
    iou_thresh: float = 0.2,
    nms_thresh: float | None = 0.8,
    mask_iou: bool = False,
) -> None:
    exp_path = ROOT / "outputs" / exp_name
    eval_dir = exp_path / f"eval-{dataset}"
    eval_dir.mkdir(exist_ok=True)

    inference_driver = create_inference_driver(
        exp_path=exp_path,
        device=torch.device("cuda"),
        is_jit_scripted=is_jit_scripted,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
    )

    cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
    tag_name_to_num_classes = cfg.INPUT.get("TAG_NAME_TO_NUM_CLASSES", {})
    tag_names = list(tag_name_to_num_classes)

    register_available_datasets(tag_names=tag_names)
    dataset_dicts = DatasetCatalog.get(f"coco-{dataset}")
    metadata = MetadataCatalog.get(f"coco-{dataset}")

    gt_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts):
        image_annotation = convert_dataset_dict_to_image_annotation(
            dataset_dict, tags_meta=tag_name_to_num_classes
        )
        gt_image_annos[image_annotation["image_id"]] = image_annotation

    pred_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts):
        image = cv2.imread(dataset_dict["file_name"])
        instances = inference_driver(
            image,
            rescale_scores_inplace=True,
            filter_by_threshold=True,
            resize_outputs_to_original_shape=True,
        )
        image_id = dataset_dict["image_id"]
        image_anno = convert_instances_to_image_annotation(
            instances, image_id, tag_name_to_num_classes
        )
        pred_image_annos[image_id] = image_anno

    metrics_per_case = {}
    for image_id, pred_anno in tqdm(pred_image_annos.items()):
        gt_anno = gt_image_annos[image_id]
        metrics = calculate_metrics(
            {image_id: pred_anno},
            {image_id: gt_anno},
            category_id_to_name_mapping={
                i: class_name for i, class_name in enumerate(metadata.thing_classes)
            },
            tags_meta=metadata.tags,
            verbose=False,
            iou_thresh=iou_thresh,
            mask_iou=mask_iou,
        )
        metrics_dict = {**metrics["per_class"], **metrics["per_tag"]}
        metrics_per_case[image_id] = {}
        for name, metrics_ in metrics_dict.items():
            metrics_per_case[image_id].update(
                {
                    f"{metric_name}/{name}": metric_value
                    for metric_name, metric_value in metrics_.items()
                }
            )
    df = pd.DataFrame(metrics_per_case).T
    df = df.round(3)
    df.to_csv(eval_dir / f"metrics_per_case_{dataset}_{iou_thresh=}_{nms_thresh=}.csv")

    metrics = calculate_metrics(
        pred_image_annos,
        gt_image_annos,
        category_id_to_name_mapping={
            i: category_name for i, category_name in enumerate(metadata.thing_classes)
        },
        tags_meta=metadata.tags,
        verbose=False,
        iou_thresh=iou_thresh,
        mask_iou=mask_iou,
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
    df = pd.Series(out)
    for metric in ["IoU", "F1", "Recall", "Precision"]:
        df[f"{metric}/mean"] = df[[i for i in df.index if i.startswith(metric)]].mean()
    df = df.round(3)
    df.to_csv(eval_dir / f"metrics_{dataset}_{iou_thresh=}_{nms_thresh=}.csv")

    for metric, metric_value in df.items():
        loguru.logger.info(f"{metric}: {metric_value:.3f}")

    for tag_name, conf_matrix in metrics["conf_matrix_per_tag"].items():
        class_names = list(metadata.tags[tag_name].values()) + ["not matched"]
        plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=class_names,
            save_path=eval_dir / f"conf_matrix_{dataset}_{tag_name}.png",
            annotate=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--nms-thresh", type=float, default=0.8)
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--jit-scripted", action="store_true")
    args = parser.parse_args()
    main(
        exp_name=args.exp_name,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        dataset=args.dataset,
        is_jit_scripted=args.jit_scripted,
    )
