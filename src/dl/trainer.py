import copy
import itertools
import logging
import os
import time
import weakref
from typing import Any, Dict, List, Set

import albumentations as A
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import torch
import torch.utils.data
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_batch_data_loader
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.common import (
    AspectRatioGroupedDataset,
    DatasetFromList,
    MapDataset,
)
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.engine import AMPTrainer, DefaultTrainer, create_ddp_model, hooks
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, DatasetEvaluators
from detectron2.modeling import build_model
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver import LRScheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules
from torch import nn

from detectron2_components.data.transforms.augmentations import (
    AlbuImageOnlyAugmentation,
)
from detectron2_ema.utils import model_ema

from .data.mapper import CustomDatasetMapper
from .evaluation.evaluator import CustomDatasetEvaluator


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg: CN):
        super(DefaultTrainer, self).__init__()

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)

        logger.info(
            f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
        )
        logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        assert cfg.SOLVER.AMP.ENABLED, "Training only in fp16 is supported."
        self._trainer = AMPTrainerWGradAcc(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            accumulate_grad_batches=cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS,
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            "trainer": weakref.proxy(self),
        }
        kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.accumulate_grad_batches = cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        self.cfg = cfg
        self._last_eval_results = None

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        if cfg.MODEL.LAZY_CONFIG is not None:
            model_cfg = LazyConfig.load(cfg.MODEL.LAZY_CONFIG)
            model = instantiate(model_cfg["model"])
            model.to(torch.device(cfg.MODEL.DEVICE))
        else:
            model = build_model(cfg)

        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))

        model_ema.may_build_model_ema(cfg, model)

        return model

    @classmethod
    def build_evaluator(
        cls, cfg: CN, dataset_name: str, output_folder: str | None = None
    ) -> DatasetEvaluator:
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators(
            [
                COCOEvaluator(dataset_name, output_dir=output_folder),
                CustomDatasetEvaluator(dataset_name),
            ]
        )

    @classmethod
    def build_train_loader(
        cls, cfg: CN
    ) -> torch.utils.data.DataLoader | AspectRatioGroupedDataset | MapDataset:
        dataset = cls.build_dataset(cfg=cfg, is_train=True)
        sampler = TrainingSampler(len(dataset))
        return build_batch_data_loader(
            dataset=dataset,
            sampler=sampler,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )

    @classmethod
    def build_test_loader(
        cls, cfg, dataset_name: str
    ) -> torch.utils.data.DataLoader | AspectRatioGroupedDataset | MapDataset:
        dataset = cls.build_dataset(cfg=cfg, is_train=False, dataset_name=dataset_name)
        sampler = InferenceSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 1, drop_last=False
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
        )
        return data_loader

    @classmethod
    def build_dataset(
        cls, cfg: CN, is_train: bool, dataset_name: str | None = None
    ) -> MapDataset:
        if is_train:
            assert dataset_name is None
            dataset_name = cfg.DATASETS.TRAIN
            filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        else:
            assert dataset_name is not None
            filter_empty = False

        dataset_dicts = get_detection_dataset_dicts(
            dataset_name, filter_empty=filter_empty
        )
        transforms = build_transforms(cfg=cfg, is_train=is_train)

        mapper = CustomDatasetMapper(
            is_train=is_train,
            transforms=transforms,
            image_format=cfg.INPUT.FORMAT,
            tag_names=list(cfg.INPUT.TAG_NAME_TO_NUM_CLASSES),
        )
        dataset = DatasetFromList(dataset_dicts, copy=False)
        dataset = MapDataset(dataset, mapper)
        return dataset

    @classmethod
    def build_lr_scheduler(
        cls, cfg: CN, optimizer: torch.optim.Optimizer
    ) -> LRScheduler:
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg: CN, model: nn.Module) -> None:
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(
            optim: torch.optim.Optimizer,
        ) -> torch.optim.Optimizer:
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            (
                model_ema.EMAHook(self.cfg, self.model)
                if cfg.MODEL_EMA.ENABLED
                else None
            ),  # add EMA hook
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
                else None
            ),
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=3
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        if cfg.MODEL_EMA.ENABLED:
            logger = logging.getLogger("detectron2")
            logger.info("Run evaluation with EMA.")
            with model_ema.apply_model_ema_and_restore(model):
                results = super().test(cfg, model, evaluators=evaluators)
        else:
            results = super().test(cfg, model, evaluators=evaluators)
        return results


class AMPTrainerWGradAcc(AMPTrainer):
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
        accumulate_grad_batches: int = 1,
    ):
        super().__init__(
            model,
            data_loader,
            optimizer,
            gather_metric_period,
            zero_grad_before_forward,
            grad_scaler,
            precision,
            log_grad_scaler,
            async_write_metrics,
        )
        self.accumulate_grad_batches = accumulate_grad_batches

    def run_step(self):
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert (
            torch.cuda.is_available()
        ), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast(dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.grad_scaler.scale(losses / self.accumulate_grad_batches).backward()

        self.after_backward()

        if (self.iter + 1) % self.accumulate_grad_batches == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            self._write_metrics(loss_dict, data_time)


def trivial_batch_collator(x):
    return x


def build_transforms(cfg, is_train: bool) -> list[T.Augmentation]:
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    transforms: list[T.Augmentation] = []
    transforms.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        transforms.extend(
            [
                AlbuImageOnlyAugmentation(
                    albu_transform=A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3
                    ),
                    prob=1.0,
                ),
                AlbuImageOnlyAugmentation(albu_transform=A.Blur(), prob=0.2),
                AlbuImageOnlyAugmentation(
                    albu_transform=A.Downscale(scale_min=0.5, scale_max=0.75), prob=0.2
                ),
                AlbuImageOnlyAugmentation(
                    albu_transform=A.MultiplicativeNoise(
                        multiplier=[0.9, 1.1], elementwise=True
                    ),
                    prob=0.1,
                ),
                AlbuImageOnlyAugmentation(
                    albu_transform=A.ImageCompression(quality_lower=50), prob=0.1
                ),
                AlbuImageOnlyAugmentation(albu_transform=A.Solarize(), prob=0.1),
            ]
        )

    return transforms
