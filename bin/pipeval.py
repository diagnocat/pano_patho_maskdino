import argparse
from typing import Literal

import cv2
import loguru
import pandas as pd
import PIL.Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from pipelines.config import ENVIRONMENT
from pipelines.graphs_2d.pano.graph_spec import build_graph_pano
from pipelines.graphs_2d.shared.utils import (
    CODE_TO_CONDITION,
    RESEARCH_TO_ATTRIBUTES_CONDITIONS,
)
from pipelines.workflows.compute import LinearComputeExecutor
from tqdm import tqdm

from src.defs import ROOT
from src.dl.data.mapper import CustomDatasetMapper
from src.dl.data.register_coco import register_available_datasets
from src.dl.evaluation.metrics import calculate_metrics
from src.dl.evaluation.utils import (
    convert_dataset_dict_to_image_annotation,
    convert_pipelines_preds_to_image_annotation,
    plot_confusion_matrix,
)


def main(
    dataset: Literal["test-orig", "test-pipelines-orig"] = "test-pipelines-orig",
    iou_thresh: float = 0.2,
    mask_iou: bool = False,
) -> None:
    eval_dir = (
        ROOT / "outputs" / f"pipelines_{ENVIRONMENT}" / f"{iou_thresh=}_{mask_iou=}"
    )
    eval_dir.mkdir(exist_ok=True, parents=True)

    register_available_datasets(
        tag_names=[
            "is_buildup",
            "post_material",
            "pbl_severity",
            "pbl_type",
            "crown_destruction",
            "involvement",
            "is_surface_distal",
            "is_surface_occlusial",
            "is_surface_lingual",
            "is_surface_mesial",
            "is_surface_vestibular",
            "is_surface_incisal",
            "is_surface_buccal",
            "is_surface_not_defined",
        ]
    )
    dataset_dicts = DatasetCatalog.get(f"coco-{dataset}")
    metadata = MetadataCatalog.get(f"coco-{dataset}")
    code_to_resarch_condition = resolve_code_to_research_condition()

    mapper = CustomDatasetMapper(
        is_train=True,
        transforms=[],
        image_format="RGB",
    )

    gt_image_annos = {}
    dataset_dicts_transformed = []
    for dataset_dict in tqdm(dataset_dicts):
        dataset_dict_transformed = mapper.apply_transforms(dataset_dict)
        image_shape = dataset_dict_transformed["image"].shape[:2]
        dataset_dict_transformed["height"] = image_shape[0]
        dataset_dict_transformed["width"] = image_shape[1]
        dataset_dicts_transformed.append(dataset_dict_transformed)

        image_annotation = convert_dataset_dict_to_image_annotation(
            dataset_dict_transformed, metadata.tags
        )
        gt_image_annos[image_annotation["image_id"]] = image_annotation

    executor = LinearComputeExecutor(
        spec=build_graph_pano(),
        verbose=False,
    )

    pred_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts_transformed):
        if "image" in dataset_dict:
            image = PIL.Image.fromarray(dataset_dict["image"])
        else:
            image = PIL.Image.fromarray(cv2.imread(dataset_dict["file_name"]))

        executor["input"] = image
        pathologies = executor["pathologies_localization"]
        pathologies = [p for p in pathologies if p["model_positive"]]
        # teeth_w_pathos = executor["assign_pathologies_to_teeth"]

        # pathologies = []
        # for tooth in teeth_w_pathos:
        #     for patho in tooth["pathologies"]:
        #         if patho["model_positive"]:
        #             pathologies.append(patho)

        image_id = dataset_dict["image_id"]
        image_anno = convert_pipelines_preds_to_image_annotation(
            pathologies=pathologies,
            image_id=image_id,
            height=dataset_dict["height"],
            width=dataset_dict["width"],
            code_to_resarch_condition=code_to_resarch_condition,
        )
        pred_image_annos[image_id] = image_anno

    metrics = calculate_metrics(
        pred_image_annos,
        gt_image_annos,
        category_id_to_name_mapping={
            i: class_name for i, class_name in enumerate(metadata.thing_classes)
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
                f"{name}/{metric_name}": metric_value
                for metric_name, metric_value in metrics_.items()
            }
        )

    df = pd.Series(out)
    for metric in ["IoU", "F1", "Recall", "Precision"]:
        df[f"{metric}/mean"] = df[[i for i in df.index if i.endswith(metric)]].mean()
    df = df.round(3)
    df.to_csv(eval_dir / "metrics.csv")

    for metric, metric_value in df.items():
        loguru.logger.info(f"{metric}: {metric_value:.3f}")

    for tag_name, conf_matrix in metrics["conf_matrix_per_tag"].items():
        class_names = list(metadata.tags[tag_name].values()) + ["not matched"]
        plot_confusion_matrix(
            conf_matrix=conf_matrix,
            class_names=class_names,
            save_path=eval_dir / f"{tag_name}.png",
            annotate=True,
        )


def resolve_code_to_research_condition() -> dict[int, str]:
    pipelines_to_research_condition = {
        v: k for k, v in RESEARCH_TO_ATTRIBUTES_CONDITIONS.items()
    }
    pipelines_to_research_condition["endo/metal_post"] = "post"
    pipelines_to_research_condition["endo/fiber_post"] = "post"

    code_to_research_condition = {}
    for code, condition_pipelines in CODE_TO_CONDITION.items():
        code_to_research_condition[code] = pipelines_to_research_condition.get(
            condition_pipelines, condition_pipelines
        )
    return code_to_research_condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-thresh", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="test-pipelines-orig")

    args = parser.parse_args()
    main(
        iou_thresh=args.iou_thresh,
        dataset=args.dataset,
    )
