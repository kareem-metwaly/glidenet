import json
import math
import os
import random
import textwrap
import typing as t
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import PIL.Image
from numpy import typing as npt
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from yarl import URL

from dataset.car.taxonomy import TAXONOMY, CSMap


# Where to store the images
BASE_S3_URL = URL("s3://scale-static-assets/cityscapes_attributes/dataset/")
BASE_HTTP_URL = URL("https://scale-static-assets.s3-us-west-2.amazonaws.com/")
CALLBACK_URL = "https://127.0.0.1/callback"

INSTRUCTION_IFRAME = """
<iframe src="https://docs.google.com/document/d/e/2PACX-1vS3dFuu8zZVybfvKApJyjg3Xh2PfrY0x05PXuk7jc03vg8EalLXFxkTqiBDvCrzYhOhDTpkFj5-yW4C/pub?embedded=true"></iframe>
"""

LIVE_API_KEY = os.environ.get("SCALE_LIVE_API_KEY", None)
TEST_API_KEY = os.environ.get("SCALE_TEST_API_KEY", None)

EncodedTuple = t.NamedTuple(
    "EncodedTuple", Category=int, Attributes=t.Optional[t.Union[int, t.List[int]]]
)

DecodedTuple = t.NamedTuple(
    "DecodedTuple", Category=str, Attributes=t.Optional[t.Mapping[str, str]]
)

cities = [
    "train/aachen",
    "train/bochum",
    "train/bremen",
    "train/cologne",
    "train/darmstadt",
    "train/dusseldorf",
    "train/erfurt",
    "train/hamburg",
    "train/hanover",
    "train/jena",
    "train/krefeld",
    "train/monchengladbach",
    "train/strasbourg",
    "train/stuttgart",
    "train/tubingen",
    "train/ulm",
    "train/weimar",
    "train/zurich",
    "val/frankfurt",
    "val/lindau",
    "val/munster",
    "test/berlin",
    "test/bielefeld",
    "test/bonn",
    "test/leverkusen",
    "test/mainz",
    "test/munich",
]


def set_cs_path(path: str) -> t.NoReturn:
    if os.environ["CITYSCAPES_DATASET"] != "":
        original = os.environ["CITYSCAPES_DATASET"]
        logger.info(f"Resetting the path of Cityscapes Dataset from {original} to {path}")
    os.environ["CITYSCAPES_DATASET"] = path
    assert cs_path() == path, "Failed to set the environment path"


def cs_path(rel_path: t.Optional[str] = None) -> str:
    assert "CITYSCAPES_DATASET" in os.environ
    path = os.environ["CITYSCAPES_DATASET"]
    if rel_path:
        path = os.path.join(path, rel_path)
    return path


def attributes_path(rel_path: t.Optional[str] = None) -> str:
    rel_path = rel_path if rel_path else ""
    rel_path = "attributes/" + rel_path
    return cs_path(rel_path)


def images_crop(
    images: t.Sequence[Image],
    bbox: t.Union["BBox", t.Tuple[float, float, float, float]],
    keep_square: bool,
) -> t.List[Image]:
    """
    Crops the images using the bounding box information. It crops a square with length depending on the longer dimension (width or height).
    Args:
        images (t.Sequence[Image]): list of images to be cropped with bbox info, they should have the same size
        bbox (t.Union["BBox", t.Tuple[float, float, float, float]]): bbox information (left, upper, width, height)
        keep_square (bool = False): used to try to preserve the cropped_image to be squared. If not set, the cropped image might be distorted when it's resized and fed to the CNN
    Returns:
        t.Tuple[Image, Image]: the cropped square image and its binary mask
    """
    if isinstance(bbox, BBox):
        bbox = bbox.as_anchor
    left, upper, width, height = bbox
    left, upper = math.floor(left), math.floor(upper)
    width, height = math.ceil(width), math.ceil(height)
    # create a square around the instance with side length
    length = max(width, height)
    image_width, image_height = images[0].size

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

    cropped = []
    for image in images:
        assert image.size == (
            image_width,
            image_height,
        ), f"{image.size} != {image_width, image_height}"
        cropped.append(image.crop((left, upper, right, lower)))
        h, w = cropped[-1].size
        if h != w and is_square:
            raise ValueError(f"output cropped image should be square, {h} & {w}")

    return cropped


@dataclass
class CARUniqueID:
    id: str  # the id format is as follows: image_path::label_path::object_id if the default delimiter is used
    delimiter: str = "::"

    def __post_init__(self):
        assert self.is_consistent

    def __repr__(self):
        return f"CARUniqueID({self.id})"

    @staticmethod
    def construct(
        image_path: str, label_path: str, object_id: t.Union[int, str], delimiter: str = "::"
    ) -> "CARUniqueID":
        image_path = (
            image_path.replace(cs_path(), "") if image_path.startswith(cs_path()) else image_path
        )
        label_path = (
            label_path.replace(cs_path(), "") if label_path.startswith(cs_path()) else label_path
        )
        idx = CARUniqueID(
            id=f"{image_path}{delimiter}{label_path}{delimiter}{object_id}", delimiter=delimiter
        )
        return idx

    @staticmethod
    def from_df(df: pd.DataFrame, append_to_df: bool = False) -> t.List["CARUniqueID"]:
        output = [
            CARUniqueID.construct(image_path=image_path, label_path=label_path, object_id=object_id)
            for image_path, label_path, object_id in zip(
                df["image_path"], df["label_path"], df["object_id"]
            )
        ]
        if append_to_df:
            df["unique_id"] = output
        return output

    def decompose(self) -> t.Mapping[str, t.Union[str, int]]:
        vals = self.id.split(self.delimiter)
        assert len(vals) == 3
        return {"image_path": vals[0], "label_path": vals[1], "object_id": int(vals[2])}

    @property
    def image_path(self) -> str:
        return self.id.split(self.delimiter)[0]

    @property
    def label_path(self) -> str:
        return self.id.split(self.delimiter)[1]

    @property
    def object_id(self) -> int:
        return int(self.id.split(self.delimiter)[2])

    @property
    def is_consistent(self) -> bool:
        image_path = self.image_path.split("/")
        label_path = self.label_path.split("/")
        image_prefix = "_".join(image_path[3].split("_")[:-1])
        label_prefix = "_".join(label_path[3].split("_")[:-2])
        return all(
            [
                image_path[1] == label_path[1],  # same split (train, val or test)
                image_path[2] == label_path[2],  # same city
                image_prefix == label_prefix,  # same prefix of the image/json name
            ]
        )

    @property
    def city(self) -> str:
        return self.image_path.split("/")[2]

    @property
    def split(self) -> str:
        return self.image_path.split("/")[1]

    @property
    def gt_type(self) -> str:
        return self.label_path.split("/")[0]

    def __eq__(self, other: "CARUniqueID") -> bool:
        return (
            (self.image_path == other.image_path)
            and (self.label_path == other.label_path)
            and (self.object_id == other.object_id)
        )


class CARPoint:
    x: int
    y: int

    def __init__(self, **kwargs):
        if "value" in kwargs:
            assert "x" not in kwargs and "y" not in kwargs
            value = kwargs["value"]
            if isinstance(value, t.Mapping):
                assert set(value.keys()) == {"x", "y"}, value.keys()
                self.x = value["x"]
                self.y = value["y"]
            elif isinstance(value, t.Sequence):
                assert len(value) == 2, value
                self.x, self.y = value
        else:
            assert set(kwargs.keys()) == {"x", "y"}, kwargs.keys()
            self.x = kwargs["x"]
            self.y = kwargs["y"]
        assert isinstance(self.x, int)
        assert isinstance(self.y, int)

    def __repr__(self) -> str:
        return f"CARPoint(x={self.x}, y={self.y})"

    @property
    def json(self) -> str:
        return json.dumps(self.as_dict)

    @staticmethod
    def from_json(text: str) -> "CARPoint":
        return CARPoint(value=json.loads(text))

    @property
    def as_dict(self) -> t.Dict[str, int]:
        return {"x": self.x, "y": self.y}

    @property
    def as_tuple(self) -> t.Tuple[int, int]:
        return self.x, self.y


class BBox:
    upper: int
    lower: int
    left: int
    right: int

    def __init__(self, **kwargs):
        possible_options = [
            {"left", "upper", "right", "lower"},
            {"left", "upper", "width", "height"},
        ]
        assert set(kwargs.keys()) in possible_options
        self.left = kwargs["left"]
        self.upper = kwargs["upper"]
        if "lower" in kwargs:
            self.lower = kwargs["lower"]
        if "right" in kwargs:
            self.right = kwargs["right"]
        if "height" in kwargs:
            self.lower = self.upper + kwargs["height"] - 1
        if "width" in kwargs:
            self.right = self.left + kwargs["width"] - 1

    @staticmethod
    def from_polygon(polygon: t.Union["CARPolygon", t.Sequence[t.Tuple[int, int]]]) -> "BBox":
        if isinstance(polygon, CARPolygon):
            polygon = polygon.as_list
        x_vals, y_vals = tuple(zip(*polygon))
        return BBox(upper=min(y_vals), lower=max(y_vals), left=min(x_vals), right=max(x_vals))

    @property
    def width(self):
        return self.right - self.left + 1

    @property
    def height(self):
        return self.lower - self.upper + 1

    @property
    def as_pil(self) -> t.Tuple[int, int, int, int]:
        """returns the bbox as tuple of (left, upper, right, lower)"""
        return self.left, self.upper, self.right, self.lower

    @property
    def as_anchor(self) -> t.Tuple[int, int, int, int]:
        """returns the bbox as tuple of (left, upper, width, height)"""
        return self.left, self.upper, self.width, self.height


class CARPolygon:
    points: t.Sequence[CARPoint]

    def __init__(
        self, points: t.Sequence[t.Union[t.Mapping[str, int], t.Tuple[int, int], CARPoint]]
    ):
        if not isinstance(points[0], CARPoint):
            points = [CARPoint(value=point) for point in points]
        self.points = points

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"CARPolygon({', '.join(str((p.x, p.y)) for p in self.points)})"

    def __getitem__(self, item: int) -> CARPoint:
        if isinstance(item, int):
            return self.points[item]
        else:
            TypeError(f"{item} has type {type(item)} which is not supported")

    def __iter__(self) -> CARPoint:
        for p in self.points:
            yield p

    @property
    def json(self) -> str:
        return json.dumps(self.as_dict)

    @staticmethod
    def from_json(text) -> "CARPolygon":
        data = json.loads(text)
        return CARPolygon(points=[CARPoint.from_json(d) for d in data])

    @property
    def as_dict(self) -> t.Sequence[t.Dict[str, int]]:
        return [p.as_dict for p in self.points]

    @property
    def as_list(self) -> t.List[t.Tuple[int, int]]:
        return [p.as_tuple for p in self.points]

    def generate_mask(self, H, W) -> Image:
        mask = PIL.Image.new("1", (W, H))
        draw = ImageDraw.Draw(mask)
        draw.polygon(self.as_list, fill=True)
        return mask

    def draw_over(
        self,
        image: Image,
        outline: t.Union[bool, t.Tuple[int, int, int, int]] = True,
        fill: t.Union[bool, t.Tuple[int, int, int, int]] = False,
    ) -> Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated, "RGBA")
        draw.polygon(self.as_list, outline=outline, fill=fill)
        return annotated

    @property
    def bbox(self) -> BBox:
        return BBox.from_polygon(self)


class CARInstance:
    unique_id: CARUniqueID
    category: str
    polygon_annotations: CARPolygon
    attributes: t.Dict[str, str]
    _meta: t.Optional[t.Mapping[str, t.Any]]
    _image: t.Optional[Image] = None

    def __init__(
        self,
        unique_id: t.Union[CARUniqueID, str],
        category: str,
        polygon_annotations: t.Optional[
            t.Union[CARPolygon, t.Sequence[t.Union[t.Mapping[str, int], CARPoint]]]
        ],
        attributes: t.Optional[t.Dict[str, t.Union[str, t.Sequence[str]]]],
        _meta: t.Optional[t.Mapping[str, t.Any]] = None,
    ):
        self.unique_id = unique_id if isinstance(unique_id, CARUniqueID) else CARUniqueID(unique_id)
        self.category = category
        self.attributes = (
            {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in attributes.items()}
            if attributes
            else None
        )
        self._meta = _meta
        if polygon_annotations:
            self.polygon_annotations = (
                polygon_annotations
                if isinstance(polygon_annotations, CARPolygon)
                else CARPolygon(polygon_annotations)
            )
        else:
            with open(cs_path(self.unique_id.label_path), "r") as f:
                data = json.load(f)
            self.polygon_annotations = CARPolygon(
                points=data["objects"][self.unique_id.object_id]["polygon"]
            )
        assert self.is_valid_types()

    def is_valid_types(self) -> bool:
        assert isinstance(self.unique_id, CARUniqueID), type(self.unique_id)
        assert isinstance(self.category, str), type(self.category)
        assert isinstance(self.polygon_annotations, CARPolygon), self.polygon_annotations
        if self.attributes is not None:
            assert isinstance(self.attributes, t.Mapping), self.attributes
            for k, v in self.attributes.items():
                assert isinstance(k, str), type(k)
                if not isinstance(v, str):
                    self.attributes.update({k: "Unclear"})
        if self._meta:
            assert isinstance(self._meta, t.Mapping)
        return True

    def __repr__(self):
        return (
            f'CARInstance(unique_id="{self.unique_id.id}", category="{self.category}", '
            f"polygon_annotations={self.polygon_annotations}, attributes={self.attributes})"
        )

    @property
    def json(self):
        return json.dumps(self.as_dict)

    @staticmethod
    def from_json(text: str) -> "CARInstance":
        data = json.loads(text)
        data["polygon_annotations"] = CARPolygon.from_json(data["polygon_annotations"]).points
        return CARInstance(**data)

    @staticmethod
    async def from_task(task: CategorizationTask) -> "CARInstance":
        await task.load_metadata()
        assert len(task.params["layers"]["polygons"]) == 1, task.params["layers"]["polygons"]
        return CARInstance(
            unique_id=task.metadata["unique_id"],
            category=task.params["layers"]["polygons"][0]["label"],
            polygon_annotations=task.params["layers"]["polygons"][0]["vertices"],
            attributes=task.response["taxonomies"] if task.response else None,
            _meta={"task_id": task._id},
        )

    @classmethod
    async def from_subtask(cls, subtask: CategorizationSubtask) -> "CARInstance":
        task = CategorizationTask(subtask.task)
        return await cls.from_task(task)

    def save(self, path: str, file_opts: str = "w"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, file_opts) as f:
            json.dump(self.json, f)

    @staticmethod
    def load(path: str) -> "CARInstance":
        with open(path, "r") as f:
            data = json.load(f)
        return CARInstance.from_json(data)

    @property
    def as_dict(self) -> t.Dict[str, t.Any]:
        out = {
            "unique_id": self.unique_id.id,
            "polygon_annotations": self.polygon_annotations.as_dict,
            "attributes": self.attributes,
            "category": self.category,
        }
        if self._meta:
            out.update({"_meta": self._meta})
        return out

    @staticmethod
    def from_dict(dct) -> "CARInstance":
        return CARInstance(**dct)

    def generate_annotations(
        self,
        font_fill: t.Union[bool, t.Tuple[int, int, int, int]] = True,
        outline: t.Union[bool, t.Tuple[int, int, int, int]] = True,
        fill: t.Union[bool, t.Tuple[int, int, int, int]] = False,
        font: ImageFont.FreeTypeFont = None,
        crop: bool = False,
    ) -> Image:
        if not font:
            font = ImageFont.truetype(
                "Pillow/Tests/fonts/FreeMono.ttf",
                60,
            )

        annotated = self.polygon_annotations.draw_over(
            self.image_nocache, outline=outline, fill=fill
        )

        if crop:
            annotated = annotated.crop(self.polygon_annotations.bbox.as_pil)

        w, h = annotated.size
        required_offset = sum(
            font.getsize(line)[1] for line in textwrap.wrap(str(self.attributes), width=w)
        )
        new_size = w, h + required_offset
        layer = PIL.Image.new("RGB", new_size, (255, 255, 255))
        layer.paste(annotated, tuple(map(lambda x: int((x[0] - x[1]) / 2), zip(new_size, (w, h)))))
        draw = ImageDraw.Draw(annotated, "RGBA")
        offset = h
        for line in textwrap.wrap(str(self.attributes), width=w):
            draw.text((0, h + offset), line, fill=font_fill, font=font)
            offset += font.getsize(line)[1]
        return annotated

    @property
    def instance_id(self) -> int:
        return self.unique_id.object_id

    @property
    def image_path(self):
        return os.path.join(cs_path(), self.unique_id.image_path)

    @property
    def label_path(self):
        return os.path.join(cs_path(), self.unique_id.label_path)

    @property
    def image(self) -> Image:
        if self._image:
            return self._image
        else:
            self._image = self.image_nocache
            return self._image

    @property
    def image_nocache(self) -> Image:
        if self._image:
            return self._image
        else:
            return PIL.Image.open(self.image_path).convert("RGB")

    @property
    def image_size(self) -> Image.size:
        return self.image_nocache.size

    @property
    def binary_mask(self) -> Image:
        w, h = self.image_size
        return self.polygon_annotations.generate_mask(H=h, W=w)

    @property
    def mask_pixel_count(self) -> int:
        return (np.asarray(self.binary_mask) == 1).sum()  # NOQA: Expected Type

    @property
    def instances_same_image(self) -> "CARInstances":
        label_path = cs_path(self.unique_id.label_path)
        with open(label_path, "r") as f:
            objects = json.load(f)["objects"]
        instances = [
            CARInstance(
                unique_id=CARUniqueID.construct(
                    image_path=self.unique_id.image_path,
                    label_path=self.unique_id.label_path,
                    object_id=i,
                ),
                category=CSMap(obj["label"]),
                polygon_annotations=[CARPoint(x=p[0], y=p[1]) for p in obj["polygon"]],
                attributes=None,
            )
            for i, obj in enumerate(objects)
        ]
        return CARInstances(instances=instances)

    def instances_matrix(self, scale: int) -> npt.ArrayLike:
        """returns 6 channels array [confidence, center_x, center_y, width, height, label]"""
        output = np.zeros([6, scale, scale])

        for instance in self.instances_same_image:
            left, upper, width, height = (
                scale * value for value in instance.polygon_annotations.bbox.as_anchor
            )
            left, width = left / self.image_size[0], width / self.image_size[0]
            upper, height = upper / self.image_size[0], height / self.image_size[1]
            width = min(width, scale)
            height = min(height, scale)
            x, y = min(left + width / 2, scale), min(upper + height / 2, scale)
            ix, iy = min(int(x), scale - 1), min(int(y), scale - 1)
            if output[0, iy, ix] == 1:
                # if we already marked this cell for an instance, compare between them and take the one with the highest area overlap
                current_x, current_y = output[1, iy, ix], output[2, iy, ix]
                current_width, current_height = output[3, iy, ix], output[4, iy, ix]
                left_overlap_current = (current_x - current_width / 2).clip(min=0)
                right_overlap_current = (current_x + current_width / 2).clip(max=1)
                upper_overlap_current = (current_y - current_height / 2).clip(min=0)
                lower_overlap_current = (current_y + current_height / 2).clip(max=1)
                left_overlap_new = max(x - ix - width / 2, 0)
                right_overlap_new = min(x - ix + width / 2, 1)
                upper_overlap_new = max(y - iy - height / 2, 0)
                lower_overlap_new = min(y - iy + height / 2, 1)
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
            output[5, iy, ix] = TaxonomyCoDec().encode(instance=(instance.category, None))[
                0
            ]  # Category Class ID
        assert np.all(np.bitwise_or(output[0] == 0, output[0] == 1)), output[0]
        assert np.all(np.bitwise_and(0 <= output[1], output[1] <= 1)), output[1]
        assert np.all(np.bitwise_and(0 <= output[2], output[2] <= 1)), output[2]
        assert np.all(np.bitwise_or(output[0] == 0, output[3] > 0)), output[3]
        assert np.all(np.bitwise_or(output[0] == 0, output[4] > 0)), output[4]
        return output

    @property
    def attributes_vectorized(self):
        codec = TaxonomyCoDec()
        return codec.encode(self, return_vector=True)[1]

    def __add__(self, other) -> "CARInstances":
        if isinstance(other, CARInstance):
            return CARInstances(instances=[self, other])
        elif isinstance(other, CARInstances):
            return CARInstances(instances=[self, *other.instances])
        else:
            raise ValueError

    def __radd__(self, other) -> t.Union["CARInstance", "CARInstances"]:
        if other == 0:
            return self
        if isinstance(other, CARInstance):
            return CARInstances(instances=[self, other])
        elif isinstance(other, CARInstances):
            return CARInstances(instances=[self, *other.instances])
        else:
            raise ValueError

    def __eq__(self, other) -> bool:
        return self.unique_id == other.unique_id and self.attributes == self.attributes


class CARInstances:
    instances: t.List[CARInstance]

    def __init__(self, instances: t.Optional[t.Iterable[CARInstance]] = None):
        self.instances = list(instances) if instances else []

    def __iter__(self) -> CARInstance:
        for instance in self.instances:
            yield instance

    def __getitem__(self, item: t.Union[int, slice]) -> t.Union[CARInstance, "CARInstances"]:
        if isinstance(item, int):
            return self.instances[item]
        elif isinstance(item, slice):
            return CARInstances(instances=[self[idx] for idx in range(*item.indices(len(self)))])
        else:
            raise TypeError(f"{item} with {type(item)} is invalid")

    def __len__(self) -> int:
        return len(self.instances)

    def __repr__(self) -> str:
        s = [
            f'{os.path.splitext(os.path.basename(instance.unique_id.label_path))[0].replace("_polygons", "")}::{instance.unique_id.object_id}'
            for instance in self
        ]
        if len(s) > 8:
            s = s[:3] + ["..."] + s[-3:]
        s = ", ".join(s)
        return f"CARInstances[{len(self)}]({s})"

    @staticmethod
    def get_instances(other: t.Union[CARInstance, "CARInstances", t.Sequence[CARInstance]]):
        if isinstance(other, CARInstance):
            other = [other]
        return other.instances if isinstance(other, CARInstances) else other

    def __add__(
        self, other: t.Union[CARInstance, "CARInstances", t.Sequence[CARInstance]]
    ) -> "CARInstances":
        return CARInstances(instances=self.instances + self.get_instances(other))

    def __radd__(
        self, other: t.Union[CARInstance, "CARInstances", t.Sequence[CARInstance]]
    ) -> "CARInstances":
        if not other:
            return self
        self.instances.extend(self.get_instances(other))
        return self

    def __eq__(self, other) -> bool:
        if len(self) == len(other):
            return all(
                self_instance == other_instance
                for self_instance, other_instance in zip(self.instances, other.instances)
            )
        return False

    @staticmethod
    def from_json(text: str) -> "CARInstances":
        instances = [CARInstance.from_dict(d) for d in text]
        return CARInstances(instances=instances)

    @property
    def json(self) -> str:
        return "[\n" + ",\n".join([instance.json for instance in self]) + "\n]"

    @staticmethod
    def load(path: str) -> "CARInstances":
        with open(path, "r") as f:
            data = json.load(f)
        return CARInstances.from_json(data)

    def save(self, path: str) -> t.NoReturn:
        with open(path, "w") as f:
            f.write(self.json)

    def images_paths(self) -> t.Set[str]:
        return set([instance.image_path for instance in self])

    def filter(
        self,
        category: t.Union[str, t.Collection[str]],
        attribute: t.Optional[t.Mapping[str, str]] = None,
    ) -> "CARInstances":
        if isinstance(category, str):
            category = [category]
        instances = (instance for instance in self if instance.category in category)

        if attribute:
            for name, value in attribute.items():
                instances = (instance for instance in instances if name in instance.attributes)
                instances = (
                    instance for instance in instances if instance.attributes[name] == value
                )

        return CARInstances(instances=instances)

    def shuffle(self, in_place: bool = False) -> "CARInstances":
        if in_place:
            random.shuffle(self.instances)
            return self
        else:
            return CARInstances(random.sample(self.instances, len(self)))

    def split_by_images(self) -> t.Dict[str, "CARInstances"]:
        images_paths = {}
        for instance in self.instances:
            if instance.image_path in images_paths:
                images_paths[instance.image_path] += instance
            else:
                images_paths[instance.image_path] = CARInstances([])
        return images_paths


class TaxonomyCoDec:
    categories_map: t.List[str]
    attributes_vector_lengths: t.Dict[str, int]
    attributes_combinations: t.Dict[str, int]
    attributes_set: t.List[str]

    def __init__(self):
        self.categories_map = [category.name for category in TAXONOMY]
        self.attributes_vector_lengths = {
            category.name: category.attributes.vector_length for category in TAXONOMY
        }
        self.attributes_combinations = {
            category.name: category.attributes.n_combinations for category in TAXONOMY
        }
        self.categories_map.append("unknown")
        self.attributes_vector_lengths.update({"unknown": 0})
        self.attributes_set = [
            f"{category.name}::{attribute.name}::{value}"
            for category in TAXONOMY
            for attribute in category.attributes
            for value in attribute.value
        ]

    @property
    def n_categories(self) -> int:
        return len(self.categories_map)

    @property
    def n_attributes(self) -> int:
        return sum(self.category_length(category) for category in self.categories_map)

    def category_length(self, category: t.Union[str, int]) -> int:
        if isinstance(category, int):
            category = self.decode(category, Attributes=None)
        return self.attributes_vector_lengths[category]

    def encode(
        self,
        instance: t.Union[CARInstance, DecodedTuple],
        return_vector: bool = False,
    ) -> EncodedTuple:
        if isinstance(instance, CARInstance):
            category_name = instance.category
            attributes = (
                {attr_name: attr_val for attr_name, attr_val in instance.attributes.items()}
                if instance.attributes
                else None
            )
        else:
            if not isinstance(instance, DecodedTuple):
                instance = DecodedTuple(*instance)
            category_name = instance.Category
            attributes = instance.Attributes
        category_idx = self.categories_map.index(category_name)

        if attributes:
            if return_vector:
                attribute_level = 0
                attributes_index = [0 for _ in range(self.attributes_vector_lengths[category_name])]
                for attribute_name, attribute_value in attributes.items():
                    attribute_taxonomy = TAXONOMY.fetch(
                        category=category_name, attribute=attribute_name
                    )
                    attribute_idx = attribute_taxonomy.value_to_index(attribute_value)
                    attributes_index[attribute_idx + attribute_level] = 1
                    attribute_level += attribute_taxonomy.n_values
            else:
                attributes_index = 0
                attributes_level = 1
                for attribute_name, attribute_value in attributes.items():
                    attribute_taxonomy = TAXONOMY.fetch(
                        category=category_name, attribute=attribute_name
                    )
                    attribute_idx = attribute_taxonomy.value_to_index(attribute_value)
                    attributes_index += attributes_level * attribute_idx
                    attributes_level *= attribute_taxonomy.n_values
        else:
            attributes_index = None
        return EncodedTuple(Category=category_idx, Attributes=attributes_index)

    def decode(
        self,
        Category: type(EncodedTuple.Category),
        Attributes: type(EncodedTuple.Attributes),
        soft_values: bool = False,
    ) -> DecodedTuple:
        """soft_values if True, the decoder will retrieve tha argmax without making sure that only one value is one"""
        category = self.categories_map[Category]
        mapping = {}
        if Attributes is not None:
            if isinstance(Attributes, int):
                attributes = TAXONOMY.fetch(category=category).attributes
                attributes_level = attributes.n_combinations
                for attribute in reversed(attributes):
                    attributes_level = attributes_level / attribute.n_values
                    assert attributes_level == int(attributes_level)
                    attribute_idx = int(Attributes / attributes_level)
                    Attributes -= attribute_idx * attributes_level
                    mapping.update({attribute.name: attribute.index_to_value(attribute_idx)})
                assert attributes_level == 1
            elif isinstance(Attributes, t.List):
                attributes = TAXONOMY.fetch(category=category).attributes
                attributes_level = attributes.vector_length
                for attribute in reversed(attributes):
                    attributes_level -= attribute.n_values
                    if soft_values:
                        # get argmax from Attributes[attributes_level:]
                        attribute_idx = max(
                            range(len(Attributes) - attributes_level),
                            key=lambda i: Attributes[attributes_level + i],
                        )
                    else:
                        attribute_idx = [
                            i for i, v in enumerate(Attributes[attributes_level:]) if v == 1
                        ]
                        assert len(attribute_idx) == 1, Attributes[attributes_level:]
                        attribute_idx = attribute_idx[0]
                    mapping.update({attribute.name: attribute.index_to_value(attribute_idx)})
                    Attributes = Attributes[:attributes_level]
                assert attributes_level == 0 and len(Attributes) == 0
            else:
                raise NotImplementedError
        return DecodedTuple(Category=category, Attributes=mapping)
