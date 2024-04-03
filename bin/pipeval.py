import cv2
import loguru
import pandas as pd
import PIL.Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from pipelines.config import ENVIRONMENT
from pipelines.graphs_2d.pano.graph_spec import build_graph_pano
from pipelines.workflows.compute import LinearComputeExecutor
from tqdm import tqdm

from src.defs import PROCESSED_DATA_PATH, ROOT
from src.dl.data.mapper import CustomDatasetMapper
from src.dl.data.register_coco import register_coco_instances_w_extra_keys
from src.dl.evaluation.metrics import calculate_metrics
from src.dl.evaluation.utils import (
    convert_dataset_dict_to_image_annotation,
    convert_pipelines_preds_to_image_annotation,
)


def main():
    register_coco_instances_w_extra_keys(
        "test",
        PROCESSED_DATA_PATH / "coco" / "annotations/instances_pipelines_val.json",
        PROCESSED_DATA_PATH / "coco" / "pipelines_val",
        extra_annotation_keys=["is_mask_annotated"],
        tag_names=[
            "tooth_num",
            "is_tooth_germ",
            "is_radix",
            "is_supernumeric",
            "impaction",
        ],
    )
    dataset_dicts = DatasetCatalog.get("test")
    metadata = MetadataCatalog.get("test")

    mapper = CustomDatasetMapper(
        is_train=True,
        transforms=[],
        image_format="RGB",
    )

    gt_image_annos = {}
    dataset_dicts_transformed = []
    for dataset_dict in tqdm(
        dataset_dicts, desc="Convert dt2 format to evaluation format"
    ):
        dataset_dict_transformed = mapper.apply_transforms(dataset_dict)
        image_shape = dataset_dict_transformed["image"].shape[:2]
        dataset_dict_transformed["height"] = image_shape[0]
        dataset_dict_transformed["width"] = image_shape[1]
        dataset_dicts_transformed.append(dataset_dict_transformed)

        image_annotation = convert_dataset_dict_to_image_annotation(
            dataset_dict_transformed, tags_meta=metadata.tags
        )
        gt_image_annos[image_annotation["image_id"]] = image_annotation

    executor = LinearComputeExecutor(
        spec=build_graph_pano(),
        verbose=False,
    )

    pred_image_annos = {}
    for dataset_dict in tqdm(dataset_dicts_transformed, desc="Running inference"):
        if "image" in dataset_dict:
            image = PIL.Image.fromarray(dataset_dict["image"])
        else:
            image = PIL.Image.fromarray(cv2.imread(dataset_dict["file_name"]))
        executor["input"] = image
        teeth_loc = executor["teeth_localization"]
        image_id = dataset_dict["image_id"]
        image_anno = convert_pipelines_preds_to_image_annotation(
            teeth_loc=teeth_loc,
            image_id=image_id,
            height=dataset_dict["height"],
            width=dataset_dict["width"],
        )
        pred_image_annos[image_id] = image_anno

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

    for metric, metric_value in out.items():
        loguru.logger.info(f"{metric}: {metric_value:.3f}")

    df = pd.Series(out)
    for metric in ["IoU", "F1", "Recall", "Precision"]:
        df[f"{metric}/mean"] = df[[i for i in df.index if i.startswith(metric)]].mean()
    df = df.round(3)
    df.to_csv(ROOT / "outputs" / f"pipelines_{ENVIRONMENT}.csv")


if __name__ == "__main__":
    main()
