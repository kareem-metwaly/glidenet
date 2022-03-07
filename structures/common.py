import typing as t
from dataclasses import dataclass

import yaml
from core_utils.serialization import deserialize, serialize

from structures.dataset import DatasetConfig
from structures.model import ModelConfig


LRConfig = t.Mapping[str, t.Optional[float]]


@dataclass
class OptimConfig:
    lr: LRConfig
    schedule: t.Optional[t.Sequence[int]] = None
    clip_grad_value: t.Optional[int] = 1
    weight_decay: t.Optional[float] = None  # will be set to 0.0001 if None


@dataclass
class S3SyncConfig:
    path: str
    sync_freq: int


@dataclass
class TrainerConfig:
    optim: OptimConfig
    s3: S3SyncConfig
    max_epochs: int
    phases_changes: t.Optional[t.Sequence[int]] = None


@dataclass
class Hyperparameters:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig

    def _log(self):
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Dataset: {self.dataset.__class__.__name__}")
        logger.info(f"configurations are {self}")

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "Hyperparameters":
        out = deserialize(Hyperparameters, d)
        out._log()
        return out

    @staticmethod
    def from_file(path: str) -> "Hyperparameters":
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return Hyperparameters.from_dict(d)

    def to_dict(self) -> t.Mapping[str, t.Any]:
        return serialize(self)
