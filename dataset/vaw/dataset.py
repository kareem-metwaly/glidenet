import hashlib
import json
import math
import os
import pickle as pkl
import typing as t
import warnings
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import requests
import sh
import torch
import torch.utils.data as thd
import torchvision.transforms as T
import tqdm
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union

from dataset.vaw.images_crawler import instantiate_crawler
from structures.abstract_classes import ABCDataset
from structures.dataset import AugmentationsConfig, VAWDatasetConfig
from structures.model import ModelInputItem, ModelInputItems


def intervals_exists(collection_intervals, value):
    for interval in collection_intervals:
        assert len(interval) == 2
        if interval[0] <= value <= interval[1]:
            return True
    return False


@dataclass
class Instance:
    bbox: t.Tuple[
        float, float, float, float
    ]  # [left, upper, width, height] scaled from 0 to 1 where 1 is full width (or height) of the image
    category_name: str
    category: torch.int

    @staticmethod
    def from_vaw_data_sample(
        instance_bbox: t.Tuple[
            float, float, float, float
        ],  # [left, upper, width, height] absolute coordinates
        category: torch.int,
        category_name: str,
        image_shape: t.Tuple[int, int],  # WxH
    ) -> "Instance":
        img_W, img_H = image_shape
        left, width = instance_bbox[0] / img_W, instance_bbox[2] / img_W
        upper, height = instance_bbox[1] / img_H, instance_bbox[3] / img_H
        return Instance(
            bbox=(left, upper, width, height),
            category=category,  # will be set later according to Labels encoder/decoder
            category_name=category_name,
        )


@dataclass
class VAWDataSample:
    image_id: int  # corresponds to the same id in VG dataset
    instance_id: int  # Unique instance ID
    instance_bbox: t.Tuple[float, float, float, float]  # [left, upper, width, height]
    instance_polygon: t.List[
        t.List[t.Tuple[float, float]]
    ]  # the first list represents the number of polygons, the second nested one contains a list of (x,y) coordinates of each polygon
    object_name: str  # Name of the object for the instance
    positive_attributes: t.List[str]  # Explicitly labeled positive attributes
    negative_attributes: t.List[str]  # Explicitly labeled negative attributes
    image_url: str
    image_local_path: t.Optional[str]  # attempt to read the image out of that path
    image_size: t.Tuple[int, int]  # Size of the original image (width x height)
    instances_in_image: t.Optional[
        t.Sequence[Instance]
    ] = None  # A list of all instances in that image
    _image: t.Optional[Image.Image] = None  # Stores the actual image, if it is cached
    _mask: t.Optional[Image.Image] = None  # Stores the binary mask of the object

    @staticmethod
    def from_dict(
        sample: t.Dict[str, t.Any],
        vg_data: t.Dict[int, t.Dict[str, t.Optional[t.Union[int, str]]]],
        attempt_local_path: t.Optional[str],
    ) -> "VAWDataSample":
        image_id = int(sample["image_id"])
        l, u, w, h = sample["instance_bbox"]
        polygon_coords = (
            sample["instance_polygon"]
            if sample["instance_polygon"]
            else [[(l, u), (l, u + h), (l + w, u + h), (l + w, u)]]
        )
        data_sample = VAWDataSample(
            image_id=image_id,
            instance_id=int(sample["instance_id"]),
            instance_bbox=(l, u, w, h),
            instance_polygon=polygon_coords,
            object_name=sample["object_name"],
            positive_attributes=sample["positive_attributes"],
            negative_attributes=sample["negative_attributes"],
            image_url=vg_data[image_id]["url"],
            image_local_path=os.path.join(attempt_local_path, str(image_id) + ".png")
            if attempt_local_path
            else None,
            image_size=(vg_data[image_id]["width"], vg_data[image_id]["height"]),
        )
        return data_sample

    @property
    def image_cache(self) -> Image.Image:
        # caches the image if retrieving online
        if not self._image:
            self._image = self.image_nocache
        return self._image

    @property
    def image_nocache(self) -> Image.Image:
        # retrieves the image if it stored or else just retrieve it without storing it
        if self._image:
            return self._image
        if self.image_local_path:
            try:
                return Image.open(self.image_local_path).convert("RGB")
            except Exception:  # the file either is corrupted or not found
                pass
        return Image.open(requests.get(self.image_url, stream=True).raw).convert("RGB")

    @property
    def bbox_area_ratio(self):
        instance_area = np.prod(self.instance_bbox[2:])  # width x height
        image_area = np.prod(self.image_size)
        return instance_area / image_area

    @property
    def polygon_area_ratio(self):
        instance_area = unary_union([Polygon(coords) for coords in self.instance_polygon]).area
        image_area = np.prod(self.image_size)
        return instance_area / image_area

    @property
    def cropped_area_ratio(self):
        instance_area = unary_union([Polygon(coords) for coords in self.instance_polygon]).area
        minx, miny, maxx, maxy = unary_union(
            [Polygon(coords) for coords in self.instance_polygon]
        ).bounds
        cropped_area = (maxx - minx + 1) * (maxy - miny + 1)
        return instance_area / cropped_area

    def get_objects_in_the_image(self, vaw_data: t.Sequence["VAWDataSample"], labels: "Labels"):
        # logger.info(f"working with {self.instance_id}")
        self.instances_in_image = [
            Instance.from_vaw_data_sample(
                instance_bbox=vaw_sample.instance_bbox,
                category=labels.cls_encode(vaw_sample.object_name),
                category_name=vaw_sample.object_name,
                image_shape=vaw_sample.image_size,
            )
            for vaw_sample in vaw_data
            if vaw_sample.image_id == self.image_id
        ]

    def instances_matrix(self, scale: int) -> torch.Tensor:
        """returns 6 channels tensor [confidence, center_x, center_y, width, height, label]"""
        assert self.instances_in_image
        output = torch.zeros([6, scale, scale])
        for instance in self.instances_in_image:
            left, upper, width, height = (torch.tensor(scale * value) for value in instance.bbox)
            width = width.clamp(max=scale)
            height = height.clamp(max=scale)
            x, y = (left + width / 2).clamp(max=scale), (upper + height / 2).clamp(max=scale)
            ix, iy = (x.floor().int()).clamp(max=scale - 1), (y.floor().int()).clamp(max=scale - 1)
            if output[0, iy, ix] == 1:
                # if we already marked this cell for an instance, compare between them and take the one with the highest area overlap
                current_x, current_y = output[1, iy, ix], output[2, iy, ix]
                current_width, current_height = output[3, iy, ix], output[4, iy, ix]
                left_overlap_current = (current_x - current_width / 2).clamp(min=0)
                right_overlap_current = (current_x + current_width / 2).clamp(max=1)
                upper_overlap_current = (current_y - current_height / 2).clamp(min=0)
                lower_overlap_current = (current_y + current_height / 2).clamp(max=1)
                left_overlap_new = (x - ix - width / 2).clamp(min=0)
                right_overlap_new = (x - ix + width / 2).clamp(max=1)
                upper_overlap_new = (y - iy - height / 2).clamp(min=0)
                lower_overlap_new = (y - iy + height / 2).clamp(max=1)
                area_current = (right_overlap_current - left_overlap_current) * (
                    lower_overlap_current - upper_overlap_current
                )
                area_new = (right_overlap_new - left_overlap_new) * (
                    lower_overlap_new - upper_overlap_new
                )
                if area_new < area_current:  # do not overwrite
                    continue
            output[0, iy, ix] = 1  # Confidence Score
            output[1, iy, ix] = x - ix  # relative center x to that cell
            output[2, iy, ix] = y - iy  # relative center y to that cell
            output[3, iy, ix] = width  # relative width
            output[4, iy, ix] = height  # relative height
            output[5, iy, ix] = instance.category  # Category Class ID
        assert (output[0] == 0).bitwise_or(output[0] == 1).all(), output[0]
        assert (0 <= output[1]).bitwise_and(output[1] <= 1).all(), output[1]
        assert (0 <= output[2]).bitwise_and(output[2] <= 1).all(), output[2]
        assert (output[0] == 0).bitwise_or(output[3] > 0).all(), output[3]
        assert (output[0] == 0).bitwise_or(output[4] > 0).all(), output[4]
        return output

    @property
    def binary_mask(self, cache: bool = False):
        if self._mask is not None:
            return self._mask
        W, H = self.image_nocache.size
        mask = Image.new("1", (W, H))
        draw = ImageDraw.Draw(mask)
        for coords in self.instance_polygon:
            draw.polygon([tuple(c) for c in coords], fill=True)
        if cache:
            self._mask = mask
        return mask

    def cropped_image(
        self,
        images: t.Optional[t.Union[Image.Image, t.Sequence[Image.Image]]] = None,
        keep_square: bool = False,
    ) -> t.List[Image.Image]:
        """
        Crops the image in self._image or image using the bounding box information. It crops a square with length depending on
        the longer dimension (width or height).
        Args:
            images (t.Optional[t.Union[Image.Image, t.Sequence[Image.Image]]]): the input image(s) to crop if it is available otherwise it fetches it from self.image
            keep_square (bool = False): used to try to preserve the cropped_image to be squared. If not set, the cropped image might be distorted when it's resized and fed to the CNN
        Returns:
            cropped (t.List[Image.Image]): the cropped square image
        """
        left, upper, width, height = self.instance_bbox
        left, upper = math.floor(left), math.floor(upper)
        width, height = math.ceil(width), math.ceil(height)
        # create a square around the instance with side length
        length = max(width, height)
        image_width, image_height = self.image_size

        if keep_square:
            # TODO: deal with images where we can't keep aspect ratio (currently, we resize - destroying aspect ratio)
            if length > min(image_width, image_height):
                is_square = False
                warnings.warn("messing with aspect ratio")
                if height > width:
                    left = 0
                    right = image_width - 1
                    lower = min(upper + height, image_height - 1)
                else:
                    upper = 0
                    lower = image_height - 1
                    right = min(left + width, image_width - 1)
            else:
                is_square = True
                lower = min(upper + length, image_height - 1)
                right = min(left + length, image_width - 1)
                # the cropped region may not yet be square so we update upper and left just in case
                upper = lower - length
                left = right - length
        else:
            is_square = False
            lower = min(upper + height, image_height)
            right = min(left + width, image_width)

        images = images if images else self.image_nocache
        if isinstance(images, PIL.Image.Image):
            images = [images]

        cropped = []
        for image in images:
            assert image.size == self.image_size, f"{image.size} != {self.image_size}"
            cropped.append(image.crop((left, upper, right, lower)))
            h, w = cropped[-1].size
            if h != w and is_square:
                raise ValueError(f"output cropped image should be square, {h} & {w}")
        return cropped


class Labels:
    """
    encodes / decodes possible labels in the dataset
    Attributes Encoding returns a tensor with values of 1, 0, -1; 1 for positive attributes, -1 for negative attributes, 0 for unlabelled attributes
    Classes Encoding returns a single value tensor with the id of the class
    """

    _atts_vals: t.Sequence[str]  # ordered set of possible attributes for an object
    _classes_vals: t.Sequence[str]  # ordered set of possible classes for an object

    def __init__(self, vaw_data: t.List[VAWDataSample]):
        attributes_values = set()
        classes_values = set()
        for vaw_sample in vaw_data:
            attributes_values = attributes_values.union(vaw_sample.positive_attributes).union(
                vaw_sample.negative_attributes
            )
            classes_values.add(vaw_sample.object_name)
        self._attrs_vals = tuple(attributes_values)
        self._classes_vals = tuple(classes_values)

    @property
    def classes(self):
        return self._classes_vals

    @property
    def attributes(self):
        return self._attrs_vals

    def attr_encode(self, pos_vals: t.List[str], neg_vals: t.List[str]) -> torch.Tensor:
        tensor = torch.zeros([len(self.attributes)])
        # fill positive attributes with 1
        for val in pos_vals:
            try:
                idx = self.attributes.index(val)
            except ValueError:
                warnings.warn("this value are not in the Labels keys")
                continue
            tensor[idx] = 1
        # fill negative attributes with -1
        for val in neg_vals:
            try:
                idx = self.attributes.index(val)
            except ValueError:
                warnings.warn("this value are not in the Labels keys")
                continue
            tensor[idx] = -1
        return tensor

    def attr_decode(self, tensor: torch.Tensor) -> t.Dict[str, t.List[str]]:
        assert len(tensor) == len(self.attributes)
        pos_ids = tensor == 1
        neg_ids = tensor == -1
        positives = [
            attr_name for ispositive, attr_name in zip(pos_ids, self.attributes) if ispositive
        ]
        negatives = [
            attr_name for isnegative, attr_name in zip(neg_ids, self.attributes) if isnegative
        ]
        return {"positive_attributes": positives, "negative_attributes": negatives}

    def cls_encode(self, value: str) -> torch.int:
        return torch.tensor(self.classes.index(value), dtype=torch.int)

    def cls_decode(self, idx: int) -> str:
        return self.classes[idx]

    def cls_one_hot(self, inp: t.Union[int, str]) -> torch.Tensor:
        if isinstance(inp, str):
            inp = self.cls_encode(inp)
        out = torch.zeros([len(self.classes)], dtype=torch.int)
        out[inp] = 1
        return out


def process_sample(input: t.Tuple["VAWDataset", VAWDataSample]):
    vaw_dataset, sample = input
    sample.get_objects_in_the_image(vaw_data=vaw_dataset.data, labels=vaw_dataset.Labels)
    return f"finished {sample.instance_id}"


class VAWDataset(ABCDataset):
    data: t.List[VAWDataSample]

    def __init__(
        self,
        config: VAWDatasetConfig,
        mode: t.Literal["train", "test", "val"],
        labels: t.Optional[
            Labels
        ] = None,  # A Label class that is used to encode/decode classes/attributes
        classes: t.Optional[
            t.Union[int, t.Sequence[str]]
        ] = None,  # if set, limit the dataset for the first n_classes
        **kwargs,
    ):
        super(VAWDataset, self).__init__(config=config, mode=mode, **kwargs)
        self.keep_square = config.keep_square
        self.path = config.path
        self.attempt_local_path = config.attempt_local_path
        hash_str = "_".join(
            (
                str(x)
                for x in (
                    config.path,
                    mode,
                    classes,
                    config.filter_small_instances_threshold,
                    config.n_samples,
                    config.attempt_local_path,
                )
            )
        )
        if self.path.startswith("s3://"):
            new_path = self.path.replace("s3://", "/tmp/")
            logger.info(f"Syncing {self.path} to {new_path}")
            os.makedirs(new_path, exist_ok=True)
            sh.aws.s3.sync(self.path, new_path)
            self.path = new_path

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
        config_load_filename = os.path.join(
            self.path, f"dataset_{hashlib.sha256(hash_str.encode('ascii')).hexdigest()}.pkl"
        )
        if os.path.exists(config_load_filename):
            with open(config_load_filename, "rb") as f:
                logger.info(f"Loading the preprocessed data from {config_load_filename}")
                data = pkl.load(f)
                for key, val in data.items():
                    self.__dict__.update({key: val})
        else:
            vaw_json = os.path.join(self.path, f"{mode}.json")
            vg_json = os.path.join(self.path, "image_data.json")
            with open(vaw_json, "r") as f:
                vaw_data = json.load(f)
            with open(vg_json, "r") as f:
                vg_data = json.load(f)

            vg_data = {
                int(vg_sample["image_id"]): {
                    "url": vg_sample["url"],
                    "width": int(vg_sample["width"]),
                    "height": int(vg_sample["height"]),
                }
                for vg_sample in vg_data
                if "url" in vg_sample.keys()
            }
            available_ids = set(vg_data.keys())
            original_length = len(vaw_data)
            data = [
                vaw_sample
                for vaw_sample in vaw_data
                if int(vaw_sample["image_id"]) in available_ids
            ]
            if len(vaw_data) < original_length:
                warnings.warn(
                    f"skipping {original_length - len(vaw_data)} values out of {original_length}"
                )
            self.data = [
                VAWDataSample.from_dict(vaw_sample, vg_data, config.attempt_local_path)
                for vaw_sample in vaw_data
            ]

            # filter samples without any labeled attributes
            self.data = [
                sample
                for sample in self.data
                if len(sample.positive_attributes + sample.negative_attributes) > 0
            ]

            if labels:
                self.Labels = labels
                # filter classes that are not defined in the given Labels class
                self.data = [
                    sample for sample in self.data if sample.object_name in self.Labels.classes
                ]
            else:
                # self.data = self.data[:6]  # used for testing a small batch propagation
                self.Labels = Labels(self.data)

            if classes:
                if isinstance(classes, int):
                    # to find the most used classes in self.data
                    classes = sorted(
                        Counter([sample.object_name for sample in self.data]).items(),
                        key=lambda key_value: key_value[1],
                        reverse=True,
                    )[:classes]
                    print(f"Picking in mode: {mode} these classes: {classes}")
                    classes = [cls[0] for cls in classes]
                self.data = [
                    vaw_sample for vaw_sample in self.data if vaw_sample.object_name in classes
                ]
                if labels is None:
                    self.Labels = Labels(self.data)

            self.classes = classes
            # plot_histogram_areas(self.data, "bbox")
            # plot_histogram_areas(self.data, "polygon")
            if config.filter_small_instances_threshold > 0:
                warnings.warn(
                    f"Filtering instances with area less than {config.filter_small_instances_threshold * 100}% of the image area"
                )
                self.data = [
                    sample
                    for sample in self.data
                    if sample.bbox_area_ratio > config.filter_small_instances_threshold
                ]

            if config.n_samples:
                self.data = self.data[: config.n_samples]

            # get objects in the image for each sample
            # _ = process_map(
            #     process_sample,
            #     [(self, sample) for sample in self.data],
            #     max_workers=4,
            #     chunksize=100,
            #     desc="Processing samples to get samples in each instance's image",
            # )
            for sample in tqdm.tqdm(self.data, desc="Getting samples in each instance's image"):
                process_sample((self, sample))

            with open(config_load_filename, "wb") as f:
                logger.info(f"Writing the preprocessed data to {config_load_filename}")
                pkl.dump(
                    {
                        "data": self.data,
                        "n_samples": len(self.data),
                        "classes": self.classes,
                        "Labels": self.Labels,
                    },
                    f,
                )
                s3_path = config_load_filename.replace("/tmp/", "s3://")
                logger.info(f"Syncing {config_load_filename} to {s3_path}")
                sh.aws.s3.cp(config_load_filename, s3_path)

        # Filter with aspect ratio
        # plot_histogram_areas(self.data, "cropped_area_ratio")
        if config.filter_cropped_area_ratio:
            warnings.warn(f"Filtering all instances within {config.filter_cropped_area_ratio}")
            original_length = len(self.data)
            self.data = [
                sample
                for sample in self.data
                if not intervals_exists(config.filter_cropped_area_ratio, sample.cropped_area_ratio)
            ]
            new_length = len(self.data)
            logger.info(
                f"Filtered instances using cropped are ratio, original_length = {original_length} and new_length = {new_length}"
            )

        if self.attempt_local_path:
            instantiate_crawler(
                mode=mode,
                args=(
                    {sample.image_id: sample.image_url for sample in self.data},
                    self.attempt_local_path,
                ),
            )

    def __getitem__(self, idx: int) -> ModelInputItem:
        for trial in range(10):
            try:
                sample = self.data[idx]
                instance_id = torch.tensor(sample.instance_id)
                image = sample.image_nocache
                attr_label = self.Labels.attr_encode(
                    pos_vals=sample.positive_attributes, neg_vals=sample.negative_attributes
                )
                cls_name = sample.object_name
                cls_id = self.Labels.cls_encode(sample.object_name)
                mask = sample.binary_mask
                cropped_image, cropped_mask = sample.cropped_image(
                    images=[image, mask], keep_square=self.keep_square
                )

                # T.ToPILImage()(torch.cat((T.ToTensor()(image), T.ToTensor()(mask).repeat([3, 1, 1])), dim=2)).show()
                # T.ToPILImage()(torch.cat((T.ToTensor()(cropped_image), T.ToTensor()(cropped_mask).repeat([3, 1, 1])), dim=2)).show()
                # print(f"Positives are {sample.positive_attributes}")
                # print(f"Negatives are {sample.negative_attributes}")

                image = self.augmentations(image)
                mask = self.augmentations_mask(mask)
                cropped_image = self.augmentations(cropped_image)
                cropped_mask = self.augmentations_mask(cropped_mask)
                output = ModelInputItem(
                    image=image,
                    mask=mask,
                    cropped_image=cropped_image,
                    cropped_mask=cropped_mask,
                    class_id=cls_id,
                    class_name=cls_name,
                    attributes_label=attr_label,
                    id=torch.tensor(idx),
                    instance_id=instance_id,
                )
                if self.config.scale:
                    output.instances_tensor = sample.instances_matrix(self.config.scale)

                return output

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

    def get_sample_by_key(self, key: str, value: t.Any):
        return [s for s in self.data if getattr(s, key) == value]

    @property
    def n_categories(self):
        return len(self.Labels.classes)

    @property
    def n_attributes(self):
        return len(self.Labels.attributes)


def plot_histogram_areas(vaw_data: t.List[VAWDataSample], mode: str = "bbox" or "polygon"):
    assert mode in {"bbox", "polygon", "cropped_area_ratio"}, mode
    if mode == "bbox":
        name = "bbox_area_ratio"
    elif mode == "polygon":
        name = "polygon_area_ratio"
    else:
        name = "cropped_area_ratio"

    areas = [getattr(sample, name) for sample in vaw_data]
    with open(f"areas_{mode}.pkl", "wb") as f:
        pkl.dump(areas, f)
    n, bins, patches = plt.hist(areas, bins=100, range=[0, 1])
    plt.show()


if __name__ == "__main__":
    config = VAWDatasetConfig(
        path="s3://scale-ml/home/kareemmetwaly/datasets/VAW_complete/VAW/",
        classes=None,
        filter_small_instances_threshold=None,
        n_samples=None,
        augmentations=AugmentationsConfig(resize=224, normalize=True),
        batch_size=1,
        concurrency=0,
        keep_square=False,
        attempt_local_path="/home/krm/vaw_images/",
        scale=28,
        filter_cropped_area_ratio=None,
    )
    dataset = VAWDataset(config=config, mode="train", labels=None, classes=None)
    collate_fn = dataset.collate_fn
    attributes_len = len(dataset.Labels.attributes)

    # dataset = torch.utils.data.Subset(
    #     dataset, indices=[i for i in reversed(range(len(dataset)))][127000:]
    # )
    data_loader = thd.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.concurrency,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    for i, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        obj = item[0]
        if obj.class_name == "floor":
            print(dataset.Labels.attr_decode(obj.attributes_label))
            obj.show()
            continue
        pass
    dataset = VAWDataset(config=config, mode="val", labels=dataset.Labels)
    collate_fn = dataset.collate_fn
    attributes_len = len(dataset.Labels.attributes)

    # dataset = torch.utils.data.Subset(
    #     dataset, indices=[i for i in reversed(range(len(dataset)))][127000:]
    # )
    data_loader = thd.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.concurrency,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    for i, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        pass
