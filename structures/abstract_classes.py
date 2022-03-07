import tempfile
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sh
import torch
import torch.utils.data as thd
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer


DATASET_CONFIG_REGISTRY = []
MODEL_CONFIG_REGISTRY = []
MODEL_INPUT_ITEM = []
MODEL_INPUT_ITEMS = []
MODEL_OUTPUT = []
DATASET = []
Loss_CONFIG_REGISTRY = []


U = t.Type["U"]

CONFIG_TYPE = t.Literal[
    "DatasetConfig",
    "ModelConfig",
    "ModelInputItem",
    "ModelInputItems",
    "ModelOutput",
    "Dataset",
    "LossConfig",
]

DatasetConfig = t.TypeVar("DatasetConfig", bound="ABCDatasetConfig")
ModelConfig = t.TypeVar("ModelConfig", bound="ABCModelConfig")
LossConfig = t.TypeVar("LossConfig", bound="ABCLossConfig")


def config_register(config_type: CONFIG_TYPE):
    def register_config(cls):
        cls = dataclass(cls)
        if config_type == "DatasetConfig":
            DATASET_CONFIG_REGISTRY.append(cls)
            cls = ABCDatasetConfig.conform(cls)
        elif config_type == "ModelConfig":
            MODEL_CONFIG_REGISTRY.append(cls)
            cls = ABCModelConfig.conform(cls)
        elif config_type == "ModelInputItem":
            MODEL_INPUT_ITEM.append(cls)
            cls = ABCModelInputItem.conform(cls)
        elif config_type == "ModelInputItems":
            MODEL_INPUT_ITEMS.append(cls)
            cls = ABCModelInputItems.conform(cls)
        elif config_type == "ModelOutput":
            MODEL_OUTPUT.append(cls)
            cls = ABCModelOutput.conform(cls)
        elif config_type == "Dataset":
            DATASET.append(cls)
            cls = ABCDataset.conform(cls)
        elif config_type == "LossConfig":
            Loss_CONFIG_REGISTRY.append(cls)
            cls = ABCLossConfig.conform(cls)
        else:
            raise ValueError(config_type)
        return cls

    return register_config


class Flags(object):
    def __init__(self, dct=None):
        if dct is None:
            dct = dict()
        self.__dict__.update(dct)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return None

    def __bool__(self):
        if self.__dict__ == {}:
            return False
        else:
            return True

    def __repr__(self):
        return self.__dict__.__str__()

    def __iter__(self):
        for item in self.__dict__:
            yield self.__getattr__(item)

    def to_dict(self):
        if self.__dict__:
            return self.__dict__
        else:
            return None


@dataclass
class ProjectConfig:
    project_id: str
    project_name: str
    customer: str
    task_type: str

    completed_at: t.Optional[str] = None
    interval: t.Optional[str] = None
    skip_empty_tasks: t.Optional[bool] = False
    ignored_classes: t.Optional[t.Sequence[str]] = None
    warehouse_name: t.Optional[t.Literal["FULL", "PUBLIC"]] = "FULL"


class ABCModelOutput(ABC):
    PreLogits: t.Optional[Tensor]

    def __init__(self, PreLogits: t.Optional[Tensor]):
        self.PreLogits = PreLogits

    @classmethod
    def load(cls, model_name, subtask_id):
        cache_path = str(PRED_OUTPUT_DIR / model_name / f"{subtask_id}.pth")
        with tempfile.NamedTemporaryFile() as f:
            # Use aws s3 because smart_open is too slow
            sh.aws.s3.cp(cache_path, f.name)
            d = torch.load(f.name)

        return cls(PreLogits=d)

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        assert "PreLogits" in other.__annotations__
        prelogits = other.__annotations__["PreLogits"]
        assert isinstance(prelogits, type(torch.Tensor)) or isinstance(
            prelogits, type(t.Literal[None])
        )
        return other


class ABCModelConfig(ABC):
    type: str
    pretrained_state_dict: str

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCLossConfig(ABC):
    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCDatasetConfig(ABC):
    batch_size: int
    concurrency: int
    classes: t.Optional[t.Union[int, t.Sequence[str]]] = None
    projects: t.Optional[t.List[ProjectConfig]] = None

    def __hash__(self):
        return hash((hash(item) for item in self.__dict__))

    @property
    @abstractmethod
    def dataset(self) -> t.Type["ABCDataset"]:
        raise NotImplementedError

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCDataset(ABC, thd.Dataset):
    data: t.Collection[t.Any]
    config: DatasetConfig
    mode: t.Literal["train", "val", "test"]

    @abstractmethod
    def __init__(self, config: DatasetConfig, mode: t.Literal["train", "val", "test"], **kwargs):
        self.config = config
        self.mode = mode
        super(ABCDataset, self).__init__()

    @abstractmethod
    def __getitem__(self, item: int) -> "ABCModelInputItem":
        ...

    def __len__(self):
        return len(self.data)

    @staticmethod
    @abstractmethod
    def collate_fn(items: t.Sequence["ABCModelInputItem"]) -> "ABCModelInputItems":
        ...

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCModelInputItem(ABC):
    metadata: t.Optional[t.Any] = None

    @abstractmethod
    def __post_init__(self):
        ...

    @abstractmethod
    def to_device(self, device: torch.device) -> "ABCModelInputItem":
        ...

    @property
    @abstractmethod
    def as_dict(self) -> t.Dict[str, t.Any]:
        ...

    @property
    @abstractmethod
    def as_batch(self) -> "ABCModelInputItems":
        ...

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCModelInputItems(ABC):
    @abstractmethod
    def __post_init__(self):
        ...

    @staticmethod
    @abstractmethod
    def collate(model_input_items: t.Sequence[ABCModelInputItem]) -> "ABCModelInputItems":
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, idx) -> ABCModelInputItem:
        ...

    def __iter__(self) -> ABCModelInputItem:
        for idx in range(len(self)):
            yield self[idx]

    @abstractmethod
    def to_device(self, device: torch.device) -> "ABCModelInputItems":
        ...

    @classmethod
    def conform(cls, other: t.Type[U]) -> t.Type[U]:
        # TODO
        return other


class ABCDataModule(ABC):
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def get_train_dataloader(self):
        pass

    @abstractmethod
    def get_validation_dataloader(self):
        pass


class ABCModel(Module, ABC):
    model: Module

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def training_step(self, batch, epoch: int):
        pass

    @abstractmethod
    def validation_step(self, batch, epoch: int):
        pass

    @abstractmethod
    def configure_optimizers(self) -> t.Tuple[Optimizer, t.Any]:
        pass
