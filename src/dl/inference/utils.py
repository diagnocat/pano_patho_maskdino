import json
from pathlib import Path

import loguru
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from ...etl.annotation import CONDITION_CLASSES
from ...maskdino.src import modeling  # noqa
from ...maskdino.src.config.inference import prepare_inference_cfg
from ...maskdino.src.config.load import load_config
from .checkpoint import prepare_inference_checkpoint
from .driver import InferenceDriver


def create_inference_driver(
    exp_path: Path,
    device: torch.device,
    is_jit_scripted: bool,
    score_thresh: float | None = None,
    nms_thresh: float | None = None,
) -> InferenceDriver:
    if is_jit_scripted:
        checkpoint_path = exp_path / "model.ts"
        model = torch.jit.load(checkpoint_path, map_location=device)
        condition_thresholds = model.thresholds
        tag_thresholds = model.tag_thresholds
        nms_thresh = model.nms_thresh
    else:
        assert (exp_path / "last_checkpoint").exists()
        with open(exp_path / "last_checkpoint", "r") as f:
            checkpoint_fname = f.read().strip()
        checkpoint_path = exp_path / f"inference_{checkpoint_fname}"
        if not checkpoint_path.exists():
            loguru.logger.info(f"Loading checkpoint {checkpoint_fname} from {exp_path}")
            prepare_inference_checkpoint(exp_path / checkpoint_fname, exp_path)
        cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
        cfg = prepare_inference_cfg(cfg)
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(str(checkpoint_path))

        if score_thresh is None:
            condition_thresholds_path = exp_path / "condition_thresholds.json"
            if not condition_thresholds_path.exists():
                raise ValueError(
                    f"score_thresh must be specified if {condition_thresholds_path} does not exist"
                )
            loguru.logger.info(
                f"Loading condition thresholds from {condition_thresholds_path}"
            )
            with open(condition_thresholds_path, "r") as f:
                condition_thresholds = json.load(f)
        else:
            loguru.logger.info(f"Using default condition threshold: {score_thresh}")
            condition_thresholds = {
                condition: score_thresh for condition in CONDITION_CLASSES
            }

        tag_thresholds_path = exp_path / "tag_thresholds.json"
        if tag_thresholds_path.exists():
            loguru.logger.info(f"Loading tag thresholds from {tag_thresholds_path}")
            with open(tag_thresholds_path, "r") as f:
                tag_thresholds = json.load(f)
        else:
            tag_thresholds = None

    loguru.logger.info(f"Successfully loaded model from {checkpoint_path}")
    model = model.to(device)
    model = model.eval()

    inference_driver = InferenceDriver(
        model=model,
        device=device,
        condition_thresholds=condition_thresholds,
        tag_thresholds=tag_thresholds,
        nms_threshold=nms_thresh,
        is_jit_scripted=is_jit_scripted,
    )
    return inference_driver
