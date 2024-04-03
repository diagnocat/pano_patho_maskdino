import argparse
from typing import Literal

import cv2
import loguru
import numpy as np
import pandas as pd
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

from src.defs import PROCESSED_DATA_PATH, ROOT
from src.dl.data.register_coco import register_coco_instances_w_extra_keys
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
    score_thresh: float,
    missing_score_thresh: float,
    dataset: Literal["train", "val", "val_orig", "pipelines_val"] = "val_orig",
    is_jit_scripted: bool = False,
) -> None:
    exp_path = ROOT / "outputs" / exp_name
    eval_dir = exp_path / f"eval-{dataset}"
    eval_dir.mkdir(exist_ok=True)

    inference_driver = create_inference_driver(
        exp_path=exp_path,
        score_thresh=score_thresh,
        missing_score_thresh=missing_score_thresh,
        device=torch.device("cuda"),
        is_jit_scripted=is_jit_scripted,
    )

    cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
    tag_name_to_num_classes = cfg.INPUT.get("TAG_NAME_TO_NUM_CLASSES", {})
    tag_names = list(tag_name_to_num_classes)

    dataset_path = PROCESSED_DATA_PATH / "coco"
    register_coco_instances_w_extra_keys(
        "test",
        dataset_path / f"annotations/instances_{dataset}.json",
        dataset_path / dataset,
        extra_annotation_keys=["is_mask_annotated"],
        tag_names=tag_names,
    )
    dataset_dicts = DatasetCatalog.get("test")
    metadata = MetadataCatalog.get("test")

    gt_image_annos = {}
    for dataset_dict in tqdm(
        dataset_dicts, desc="Convert dt2 format to evaluation format"
    ):
        image_annotation = convert_dataset_dict_to_image_annotation(
            dataset_dict, tags_meta=tag_name_to_num_classes
        )
        gt_image_annos[image_annotation["image_id"]] = image_annotation

    pred_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts, desc="Running inference"):
        if "image" in dataset_dict:
            image = dataset_dict["image"]
        else:
            image = cv2.imread(dataset_dict["file_name"])
        instances = inference_driver(image, postprocess=True, crop_working_field=True)
        image_id = dataset_dict["image_id"]
        image_anno = convert_instances_to_image_annotation(
            instances, image_id, tag_name_to_num_classes
        )
        pred_image_annos[image_id] = image_anno

    metrics_per_case = {}
    for image_id, pred_anno in tqdm(
        pred_image_annos.items(), desc="Calculating metrics per case"
    ):
        gt_anno = gt_image_annos[image_id]
        metrics = calculate_metrics(
            {image_id: pred_anno},
            {image_id: gt_anno},
            class_id_to_name_mapping={
                i: class_name for i, class_name in enumerate(metadata.thing_classes)
            },
            tags_meta=metadata.tags,
            verbose=False,
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
    df.to_csv(
        eval_dir
        / f"metrics_per_case_{dataset}_{score_thresh=}_{missing_score_thresh=}.csv"
    )

    metrics = calculate_metrics(
        pred_image_annos,
        gt_image_annos,
        class_id_to_name_mapping={
            i: class_name for i, class_name in enumerate(metadata.thing_classes)
        },
        tags_meta=metadata.tags,
        verbose=False,
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
    df.to_csv(
        eval_dir / f"metrics_{dataset}_{score_thresh=}_{missing_score_thresh=}.csv"
    )

    for metric, metric_value in out.items():
        loguru.logger.info(f"{metric}: {metric_value:.3f}")

    for tag_name, conf_matrix in metrics["conf_matrix_per_tag"].items():
        class_names = list(metadata.tags[tag_name].values()) + ["not matched"]
        plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=class_names,
            save_path=eval_dir / f"conf_matrix_{dataset}_{tag_name}.png",
            annotate=tag_name != "tooth_num",
        )
        conf_matrix_log = np.log(conf_matrix + np.e).round() - 1
        plot_confusion_matrix(
            conf_matrix=conf_matrix_log,
            class_names=class_names,
            save_path=eval_dir / f"conf_matrix_log_{dataset}_{tag_name}.png",
            annotate=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--missing-score-thresh", type=float, default=0.3)
    parser.add_argument("--dataset", type=str, default="val_orig")
    parser.add_argument("--jit-scripted", action="store_true")
    args = parser.parse_args()
    main(
        args.exp_name,
        args.score_thresh,
        args.missing_score_thresh,
        args.dataset,
        args.jit_scripted,
    )
