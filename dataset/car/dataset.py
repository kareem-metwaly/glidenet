import os
import typing as t

import sh
import torch
import torchvision.transforms as T
from tqdm import tqdm

from dataset.car.taxonomy import TAXONOMY
from dataset.car.utils import (
    CARInstance,
    CARInstances,
    DecodedTuple,
    TaxonomyCoDec,
    attributes_path,
    cs_path,
    images_crop,
)
from structures.abstract_classes import ABCDataset
from structures.dataset import AugmentationsConfig, CARDatasetConfig
from structures.model import ModelInputItem, ModelInputItems


class CARDataset(ABCDataset):
    data: CARInstances
    codec: TaxonomyCoDec

    def __init__(
        self,
        config: CARDatasetConfig,
        mode: t.Literal["train", "val", "test"],
        **kwargs,
    ):
        super(CARDataset, self).__init__(config, mode, **kwargs)
        self.keep_square = config.keep_square
        self.path = config.path if config.path else cs_path()
        if self.path != cs_path():
            os.environ["CITYSCAPES_DATASET"] = self.path

        self.augmentations = []
        self.augmentations_mask = []
        if config.augmentations is not None:
            if config.augmentations.resize is not None:
                self.augmentations.append(
                    T.Resize(size=(config.augmentations.resize, config.augmentations.resize))
                )
                self.augmentations_mask.append(
                    T.Resize(size=(config.augmentations.resize, config.augmentations.resize))
                )
            self.augmentations.append(T.ToTensor())
            self.augmentations_mask.append(T.ToTensor())
            if config.augmentations.normalize:
                self.augmentations.append(
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                )
        else:
            self.augmentations.append(T.ToTensor())
            self.augmentations_mask.append(T.ToTensor())

        self.augmentations = T.Compose(self.augmentations)
        self.augmentations_mask = T.Compose(self.augmentations_mask)

        car_json_path = attributes_path(f"{mode}.json")
        if not os.path.exists(car_json_path):
            assert config.s3_path
            logger.info(f"Syncing {config.s3_path} to {self.path}")
            os.makedirs(self.path, exist_ok=True)
            sh.aws.s3.sync(config.s3_path, self.path)

        self.data = CARInstances.load(car_json_path)
        self.codec = TaxonomyCoDec()

        # eliminating samples without attributes
        eliminated = []
        for cat in TAXONOMY:
            if len(cat.attributes) == 0:  # doesn't have attributes
                eliminated.append(cat.name)
        eliminated = {"class_name": eliminated}
        self.data = CARInstances(
            sample for sample in self.data if sample.category not in eliminated["class_name"]
        )

        if config.n_samples:
            self.data = self.data[: config.n_samples]

    def __getitem__(self, idx: int) -> ModelInputItem:
        for trial in range(10):
            try:
                return self.from_data_sample(self.data[idx], idx)

            except Exception as err:
                print(f"idx is {idx}")
                print(f"Error is {err}")
                if trial < 9:
                    continue
                else:
                    raise err

    @staticmethod
    def collate_fn(batch: t.List[ModelInputItem]) -> ModelInputItems:
        return ModelInputItems.collate(batch)

    @property
    def n_categories(self):
        return self.codec.n_categories

    @property
    def n_attributes(self):
        return self.codec.n_attributes

    def from_data_sample(
        self, sample: CARInstance, idx: t.Optional[int] = None, metadata: t.Optional[t.Any] = None
    ) -> ModelInputItem:
        instance_id = torch.tensor(sample.instance_id)
        image = sample.image_nocache
        cls_name = sample.category
        cls_id, attr_vector = self.codec.encode(sample, return_vector=True)
        mask = sample.binary_mask
        cropped_image, cropped_mask = images_crop(
            images=[image, mask],
            bbox=sample.polygon_annotations.bbox,
            keep_square=self.keep_square,
        )

        image = self.augmentations(image)
        mask = self.augmentations_mask(mask)
        cropped_image = self.augmentations(cropped_image)
        cropped_mask = self.augmentations_mask(cropped_mask)
        output = ModelInputItem(
            image=image,
            mask=mask,
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            class_id=torch.tensor(cls_id),
            class_name=cls_name,
            attributes_label=torch.Tensor(attr_vector) if attr_vector else None,
            id=torch.tensor(idx) if idx is not None else None,
            instance_id=instance_id,
        )
        if self.config.scale:
            output.instances_tensor = torch.from_numpy(
                sample.instances_matrix(self.config.scale)
            ).to(dtype=torch.float)

        if metadata:
            output.metadata = metadata
        return output

    def prelogits_to_prediction(self, category: int, prelogits: torch.Tensor) -> DecodedTuple:
        """Converts the output of a model to the corresponding attributes name, value pair"""
        return self.codec.decode(
            Category=category, Attributes=prelogits.cpu().tolist(), soft_values=True
        )


if __name__ == "__main__":
    configs = CARDatasetConfig(
        type="CAR",
        path="/app/datasets/cityscapes/",
        s3_path="s3://scale-ml/home/kareemmetwaly/datasets/cityscapes/",
        batch_size=2,
        concurrency=0,
        n_samples=None,
        augmentations=AugmentationsConfig(resize=224, normalize=True),
        keep_square=False,
        scale=28,
    )
    data = CARDataset(config=configs, mode="train")
    problems = []
    for i in tqdm(reversed(range(len(data))), total=len(data)):
        try:
            item = data[i]
        except:
            try:
                smp = data.data[i]
                print(smp)
                item = data[i]
            except Exception as e:
                print(e)
                problems.append(i)
    print(problems)
