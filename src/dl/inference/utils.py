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
        loguru.logger.info("Loading optimized thresholds from scripted model")
        thresholds = model.thresholds
        loguru.logger.info("Loading nms threshold from scripted model")
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

        thresholds_path = exp_path / "thresholds.json"
        if score_thresh is None and not thresholds_path.exists():
            raise ValueError(
                f"score_thresh must be specified if {thresholds_path} does not exist"
            )

        if score_thresh is not None:
            loguru.logger.info(f"Using default threshold: {score_thresh}")
            thresholds = {condition: score_thresh for condition in CONDITION_CLASSES}
        else:
            loguru.logger.info(f"Loading optimized thresholds from {thresholds_path}")
            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)

    loguru.logger.info(f"Successfully loaded model from {checkpoint_path}")
    model = model.to(device)
    model = model.eval()

    inference_driver = InferenceDriver(
        model=model,
        device=device,
        probability_thresholds=thresholds,
        nms_threshold=nms_thresh,
        is_jit_scripted=is_jit_scripted,
    )
    return inference_driver
