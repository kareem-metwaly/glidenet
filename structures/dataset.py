import typing as t
from dataclasses import dataclass

from structures.abstract_classes import ABCDatasetConfig, config_register


@dataclass
class AugmentationsConfig:
    resize: t.Optional[int]  # the final shape of the cropped image will be resize x resize
    normalize: bool = True  # normalize based on pretrained pytorch models with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]


@config_register("DatasetConfig")
class VAWDatasetConfig(ABCDatasetConfig):
    type: str  # should hold value "VAW"
    path: str  # should contain 4 files; train.json, val.json, test.json, image_data.json
    # batch_size is per GPU (will be multiplied by number of GPUs)
    batch_size: int
    # concurrency is per sample (will be multiplied by batch_size)
    concurrency: int
    keep_square: bool  # when cropping should we try to keep aspect ratio while making it square
    augmentations: t.Optional[AugmentationsConfig]
    # classes if not None, bound the number of classes to this value, takes the first classes from the dataset in int,
    # or selects the classes from that list of str
    classes: t.Optional[t.Union[int, t.Sequence[str]]] = None
    filter_small_instances_threshold: t.Optional[float] = 0.0
    attempt_local_path: t.Optional[
        str
    ] = None  # a local path where to look for the image before fetching online
    n_samples: t.Optional[
        int
    ] = None  # set the size of the dataset (limiting the length of the dataset to that size)
    scale: t.Optional[int] = None  # if set the output contains
    filter_cropped_area_ratio: t.Optional[
        t.Sequence[t.Tuple[float, float]]
    ] = None  # remove instances with cropped area ratio in one of the intervals described by a tuple in the list

    def __post_init__(self):
        assert self.type == "VAW"

    @property
    def dataset(self) -> t.Type["VAWDataset"]:  # NOQA F821
        from dataset.vaw.dataset import VAWDataset

        return VAWDataset


@config_register("DatasetConfig")
class CARDatasetConfig(ABCDatasetConfig):
    type: str  # should hold value "CAR"
    path: t.Optional[str]  # should contain root path for Cityscapes
    # batch_size is per GPU (will be multiplied by number of GPUs)
    batch_size: int
    # concurrency is per sample (will be multiplied by batch_size)
    concurrency: int
    keep_square: bool  # when cropping should we try to keep aspect ratio while making it square
    augmentations: t.Optional[AugmentationsConfig]
    s3_path: t.Optional[str] = None  # Optional path to fetch data from if doesn't exist locally
    n_samples: t.Optional[
        int
    ] = None  # set the size of the training/val/test datasets (limiting the length of the dataset to that size)
    scale: t.Optional[int] = None  # if set the output contains

    def __post_init__(self):
        assert self.type == "CAR"
        self.__dict__.update({"classes": None})  # for backward compatibility

    @property
    def dataset(self) -> t.Type["CARDataset"]:  # NOQA F821
        from dataset.car.dataset import CARDataset

        return CARDataset


DatasetConfig = t.Union[CARDatasetConfig, VAWDatasetConfig]
