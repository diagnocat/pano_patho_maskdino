import json
import os
import random
import warnings
from pathlib import Path

import detectron2.utils.comm as comm
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger

import src.maskdino.src.modeling  # noqa [Register custom components]
from detectron2_ema.utils import model_ema
from src.dl.data.register_coco import register_available_datasets
from src.dl.inference.checkpoint import prepare_inference_checkpoint
from src.dl.trainer import Trainer
from src.maskdino.src.config.load import load_config
from src.maskdino.src.config.maskdino import add_maskdino_config
from src.viz import plot_and_save_from_dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(args):
    if args.eval_only:
        opts = dict(zip(args.opts[0::2], args.opts[1::2]))
        cfg = load_config(config_filepath=str(Path(opts["OUTPUT_DIR"]) / "config.yaml"))
    else:
        cfg = setup(args)

    tag_names = list(cfg.INPUT.get("TAG_NAME_TO_NUM_CLASSES", {}))
    register_available_datasets(tag_names=tag_names)

    # We need to call the registered dataset
    # to obtain full metadata
    for dataset_name in cfg.DATASETS.TRAIN:
        _ = DatasetCatalog.get(dataset_name)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        with open(Path(cfg.OUTPUT_DIR) / "last_checkpoint", "r") as f:
            checkpoint_fname = f.read().strip()
        checkpointer.load(str(Path(cfg.OUTPUT_DIR) / checkpoint_fname))
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        out = {k: v for k, v in res.items() if isinstance(v, float)}
        with open(Path(cfg.OUTPUT_DIR) / "inference" / "metrics.json", "w") as f:
            json.dump(out, f)

        return res

    if comm.is_main_process():
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        # Plot training images
        train_dataset = Trainer.build_dataset(cfg=cfg, is_train=True)
        plot_and_save_from_dataset(
            dataset=train_dataset,
            save_dir=str(Path(cfg.OUTPUT_DIR) / "before_train_viz"),
            n_save_pics=10,
            metadata=metadata,
        )

        # init wandb
        if not os.getenv("DISABLE_WANDB", False):
            project_name = "maskdino_pano_patho"
            run_name = str(cfg.OUTPUT_DIR).split("/")[-1]
            wandb.init(
                project=project_name,
                name=run_name,
                sync_tensorboard=True,
            )
    comm.synchronize()

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    out = trainer.train()
    if comm.is_main_process():
        prepare_inference_checkpoint(
            Path(cfg.OUTPUT_DIR) / "model_final.pth", Path(cfg.OUTPUT_DIR)
        )
    return out


def setup(args):
    cfg = get_cfg()
    add_maskdino_config(cfg)
    model_ema.add_model_ema_configs(cfg)
    cfg.MODEL.LAZY_CONFIG = None
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)

    if getattr(args, "opts", None) is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino"
    )

    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)

    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    print("args = ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
