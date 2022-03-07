import typing as t
import warnings
from time import time

import torch
from torch import nn

import models
from structures.abstract_classes import ABCModel
from structures.common import Hyperparameters
from structures.loss import GeneralLoss
from structures.model import ModelInputItems
from trainer.callbacks import MLFlowLogger, TensorboardLogger
from trainer.utils import rank_zero_only


class SubtaskModel(ABCModel):
    def __init__(
        self,
        hparams: Hyperparameters,
        classes: t.List[str],
        attributes: t.List[str],
        n_iters_epoch: t.Optional[int] = None,
    ):
        super().__init__()
        self.hparams = hparams
        self.n_iters_epoch = n_iters_epoch
        self.model = self.get_model(classes, attributes)
        self.train_cache = {}
        self.validation_cache = {}
        self.train_classes = set()
        self.validation_classes = set()
        self.phase_number = None
        momentum = 0.9
        tolerance = 3
        self.train_loss_tracker = {
            "current": 0,
            "momentum": momentum,
            "tolerance": tolerance,
        }  # used to check if one sample has an abnormal loss value
        self.validation_loss_tracker = {
            "current": 0,
            "momentum": momentum,
            "tolerance": tolerance,
        }  # used to check if one sample has an abnormal loss value
        self.train_accum = []  # used to log troubling instances when using single sample/batch
        self.validation_accum = []  # used to log troubling instances when using single sample/batch

        self.epoch = 0

    @property
    def inner_model(self):
        if isinstance(self.model, torch.nn.parallel.distributed.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    def get_model(self, classes: t.List[str], attributes: t.List[str]) -> nn.Module:
        try:
            model = getattr(models, self.hparams.model.type)
            return model(self.hparams.model, classes, attributes)
        except ModuleNotFoundError:
            raise NotImplementedError(f"This model is not implemented {self.hparams.model.type}")

    def forward(self, model_items: ModelInputItems) -> GeneralLoss:
        losses = self.inner_model.forward(model_items, phase_number=self.phase_number)
        return losses

    def inference(self, model_items: ModelInputItems) -> torch.Tensor:
        return self.inner_model.inference(model_items)

    def features(self, model_items: ModelInputItems) -> t.Dict[str, torch.Tensor]:
        return self.inner_model.features(model_items)

    @staticmethod
    def loss(losses: GeneralLoss) -> torch.Tensor:
        # taking into considerations the loss values only
        return losses.value

    def update_phase_number(self, epoch: int) -> int:
        changes = self.hparams.trainer.phases_changes
        for i, next_th in enumerate(changes):
            if epoch < next_th:
                self.phase_number = i + 1
                return self.phase_number
        self.phase_number = len(changes) + 1
        return self.phase_number

    def mode_step(self, model_items: ModelInputItems, mode: str, epoch: int):
        assert mode in {"train", "validation"}, mode
        losses = self.forward(model_items)
        loss = self.loss(losses)
        logs = {"loss": loss}
        # update current loss tracker
        loss_tracker = self.__getattribute__(f"{mode}_loss_tracker")
        loss_tracker["current"] = (
            loss_tracker["momentum"] * loss_tracker["current"]
            + (1 - loss_tracker["momentum"]) * loss
        )

        if loss > loss_tracker["current"] * (1 + loss_tracker["tolerance"]):
            instance_id = model_items.instance_ids
            if instance_id.shape[0] == 1:  # if we are only using a single sample/batch
                instance_id = instance_id.squeeze().item()
                warnings.warn(f"This {mode} batch has abnormal loss value, {instance_id}")
                self.__getattribute__(f"{mode}_accum").append(instance_id)
                logs.update({"troubling_instances": instance_id})

        logs.update({"loss_tracker": loss_tracker["current"]})
        if model_items.instance_ids.shape[0] == 1:
            logs.update({"ids": model_items.instance_ids.squeeze()})

        logs.update(losses.logs)
        logs = {k: v.item() if hasattr(v, "item") else v for k, v in logs.items()}
        logs = {f"{mode}/{k}": v for k, v in logs.items()}
        cache = self.__getattribute__(f"{mode}_cache")
        classes = self.__getattribute__(f"{mode}_classes")
        if losses.cache:
            for k, v in losses.cache.items():
                if k in cache:
                    cache[k].append(v)
                else:
                    cache[k] = [v]
            classes.update(model_items.classes_set)
        return loss, logs

    def training_step(self, model_items: ModelInputItems, epoch: int):
        return self.mode_step(model_items, mode="train", epoch=epoch)

    def validation_step(self, model_items: ModelInputItems, epoch: int):
        return self.mode_step(model_items, mode="validation", epoch=epoch)

    @rank_zero_only
    def on_mode_epoch_end(self, mode: str, epoch: int, trainer: "Trainer"):  # NOQA F821
        assert mode in {"train", "validation"}, mode
        if mode == "train":
            self.epoch += 1
        cache = self.__getattribute__(f"{mode}_cache")
        classes = self.__getattribute__(f"{mode}_classes")

        records = self.inner_model.on_mode_epoch_end(mode=mode, cache=cache, classes=classes)

        if records:
            for trainer_logger in trainer.loggers:
                if isinstance(trainer_logger, TensorboardLogger):
                    for k, v in records.items():
                        trainer_logger.writer.add_scalar(k, v, epoch)
                elif isinstance(trainer_logger, MLFlowLogger):
                    timestamp_ms = int(time() * 1000)
                    for k, v in records.items():
                        trainer_logger.client.log_metric(
                            trainer_logger.run_id, k, v, timestamp=timestamp_ms, step=epoch
                        )
        setattr(self, f"{mode}_cache", {})
        setattr(self, f"{mode}_classes", set())

    def on_train_epoch_end(self, epoch: int, trainer: "Trainer"):  # NOQA F821
        self.on_mode_epoch_end(mode="train", epoch=epoch, trainer=trainer)

    def on_validation_epoch_end(self, epoch: int, trainer: "Trainer"):  # NOQA F821
        self.on_mode_epoch_end(mode="validation", epoch=epoch, trainer=trainer)

    def configure_optimizers(self):
        assert "base" in self.hparams.trainer.optim.lr.keys(), self.hparams.trainer.optim.lr
        params_lists = {
            name: {
                "lr": lr_value,
                "params": [],
            }
            for name, lr_value in self.hparams.trainer.optim.lr.items()
            if lr_value is not None
        }

        for name, p in self.named_parameters():
            for base_name in params_lists.keys():
                if base_name in name:
                    params_lists[base_name]["params"].append((name, p))
                    break
            else:
                params_lists["base"]["params"].append((name, p))

        # confirming that we are considering all params
        params_bases = [
            list(zip(*params_lists[base_name]["params"])) for base_name in params_lists.keys()
        ]
        params_bases = [item[0] for item in params_bases if len(item) > 0]
        for name, p in self.named_parameters():
            for params_base in params_bases:
                if name in params_base:
                    break
            else:
                raise ValueError(f"Couldn't fine the value for {name}")

        weight_decay = (
            self.hparams.trainer.optim.weight_decay
            if self.hparams.trainer.optim.weight_decay
            else 0.0001
        )

        optimizers = [
            torch.optim.Adam(
                list(zip(*value["params"]))[1],
                lr=value["lr"],
                weight_decay=weight_decay,
            )
            for name, value in params_lists.items()
            if value["params"]
        ]

        steps = [
            x for x in self.hparams.trainer.optim.schedule if x <= self.hparams.trainer.max_epochs
        ]
        if len(steps) != len(self.hparams.trainer.optim.schedule):
            warnings.warn(
                "scheduler contains values larger than maximum iterations."
                "These values will be ignored."
            )

        schedulers = [torch.optim.lr_scheduler.MultiStepLR(optim, steps) for optim in optimizers]

        return optimizers, schedulers
