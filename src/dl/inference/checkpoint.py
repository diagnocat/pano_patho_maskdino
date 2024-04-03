from pathlib import Path

import loguru
import torch


def prepare_inference_checkpoint(
    checkpoint_path: Path,
    out_path: Path,
    verbose: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location=device)
    out = {}

    if "ema_state" in ckpt:
        out = ckpt["ema_state"]
        out.pop("pixel_mean")
        out.pop("pixel_std")
    else:
        out = ckpt["model"]
    out.pop("criterion.empty_weight")

    out_checkpoint_path = out_path / f"inference_{checkpoint_path.name}"
    torch.save({"model": out}, out_checkpoint_path)

    if verbose:
        loguru.logger.info(
            f"Successfully saved inference checkpoint: {out_checkpoint_path}"
        )
    return out_checkpoint_path
