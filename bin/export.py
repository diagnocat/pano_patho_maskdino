from __future__ import annotations

import argparse
import json
import time

import loguru
import torch
import torch._C
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from src.defs import ROOT
from src.dl.inference.checkpoint import prepare_inference_checkpoint
from src.maskdino.src import modeling  # noqa
from src.maskdino.src.config.inference import prepare_inference_cfg
from src.maskdino.src.config.load import load_config


def main(exp_name: str, nms_thresh: float = 0.8, n_test_runs: int = 10) -> None:
    exp_path = ROOT / "outputs" / exp_name

    checkpoint_path = exp_path / "inference_model_final.pth"
    if not checkpoint_path.exists():
        prepare_inference_checkpoint(exp_path / "model_final.pth", exp_path)

    cfg = load_config(config_filepath=str(exp_path / "config.yaml"))
    cfg = prepare_inference_cfg(cfg)

    model = build_model(cfg)
    model = model.eval()
    thresholds_path = exp_path / "thresholds.json"
    assert thresholds_path.exists()
    with open(thresholds_path, "r") as f:
        thresholds = json.load(f)
    model.thresholds = thresholds
    model.nms_thresh = nms_thresh

    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(str(checkpoint_path))

    model = model.cpu()
    device = torch.device("cuda")

    ts_model = torch.jit.script(model)
    torch.jit.save(ts_model, exp_path / "model.ts")
    loguru.logger.info(f"Saved to {exp_path / 'model.ts'}")

    loaded_ts_model = torch.jit.load(exp_path / "model.ts", map_location=device)

    model = model.to(device)
    for i in range(n_test_runs):
        image = torch.randn((3, 384, 512)).to(device)
        inputs = [
            {
                "image": image,
                "height": torch.as_tensor(512),
                "width": torch.as_tensor(768),
            }
        ]
        with torch.no_grad():
            start = time.time()
            preds_torch = model(inputs)[0]
            loguru.logger.info(f"PyTorch {i} {time.time() - start:.3f}")

            start = time.time()
            with torch.jit.optimized_execution(False):
                preds_ts = loaded_ts_model(inputs)[0]
            loguru.logger.info(f"TorchScript {i} {time.time() - start:.3f}")

            for field, value in preds_torch.items():
                value_predictor = preds_ts[field]

                assert torch.allclose(value, value_predictor), field


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--nms-thresh", type=float, default=0.8)
    args = parser.parse_args()
    main(args.exp_name, args.nms_thresh)
