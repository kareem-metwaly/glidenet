import typing as t
from dataclasses import dataclass
from enum import Enum

import torch
from core_utils.serialization import deserialize
from torch import Tensor
from torchvision.transforms import ToPILImage

from structures.abstract_classes import (
    ABCLossConfig,
    ABCModelConfig,
    ABCModelInputItem,
    ABCModelInputItems,
    ABCModelOutput,
    Flags,
    config_register,
)
from structures.loss import VAWLossesConfig


@config_register("ModelInputItem")
class ModelInputItem(ABCModelInputItem):
    image: Tensor
    mask: Tensor
    cropped_image: Tensor
    cropped_mask: Tensor
    class_id: Tensor
    class_name: str
    attributes_label: t.Optional[Tensor]
    id: t.Optional[Tensor]
    instance_id: Tensor
    instances_tensor: t.Optional[Tensor] = None
    metadata: t.Optional[t.Any] = None

    def __post_init__(self):
        assert (
            len(self.cropped_mask.shape) == len(self.cropped_image.shape) == 3
        ), f"The cropped image and mask should be a 3D Tensor, got {self.cropped_image.shape} and {self.cropped_mask.shape}"
        assert (
            self.image.shape[-2:] == self.mask.shape[-2:]
        ), f"The size of the mask ({self.mask.shape}) and image ({self.image.shape}) should be equal"
        assert self.mask.shape[0] == 1
        assert self.image.shape[0] == 3
        assert (
            self.cropped_image.shape[-2:] == self.cropped_mask.shape[-2:]
        ), f"The size of the cropped mask ({self.cropped_mask.shape}) and cropped image ({self.cropped_image.shape}) should be equal"
        assert (
            self.cropped_mask.shape[0] == 1
        ), f"The size of the channels of the mask should be 1, got {self.cropped_mask.shape}"
        if self.attributes_label is not None:
            assert (
                len(self.attributes_label.shape) == 1
            ), f"the attributes label should be a single dimensional tensor, got {self.attributes_label.shape}"
        if self.instances_tensor is not None:
            assert self.instances_tensor.shape[0] == 6, self.instances_tensor.shape
            assert len(self.instances_tensor.shape) == 3, self.instances_tensor.shape

    def to_device(self, device: torch.device):
        return ModelInputItem(
            image=self.image.to(device),
            mask=self.mask.to(device),
            class_name=self.class_name,
            cropped_image=self.cropped_image.to(device),
            cropped_mask=self.cropped_mask.to(device),
            class_id=self.class_id.to(device),
            attributes_label=self.attributes_label.to(device)
            if self.attributes_label is not None
            else None,
            id=self.id.to(device) if self.id is not None else None,
            instance_id=self.instance_id.to(device),
            instances_tensor=self.instances_tensor.to(device)
            if self.instances_tensor is not None
            else None,
            metadata=self.metadata,
        )

    def show(self) -> None:
        ToPILImage()(self.image).show()

    @property
    def as_dict(self) -> t.Dict[str, t.Any]:
        return {
            "image": self.image.cpu().data.numpy(),
            "cropped_image": self.cropped_image.cpu().data.numpy(),
            "mask": self.mask.cpu().data.numpy(),
            "cropped_mask": self.cropped_mask.cpu().data.numpy(),
            "class_id": self.class_id.item(),
            "class_name": self.class_name,
            "attributes_label": self.attributes_label.cpu().data.numpy()
            if self.attributes_label is not None
            else None,
            "id": self.id.item() if self.id is not None else None,
            "instance_id": self.instance_id.item(),
            "instances_tensor": self.instances_tensor.cpu().data.numpy()
            if self.instances_tensor is not None
            else None,
            "metadata": self.metadata,
        }

    @property
    def as_batch(self) -> "ModelInputItems":
        return ModelInputItems.collate([self])


@config_register("ModelInputItems")
class ModelInputItems(ABCModelInputItems):
    images: Tensor
    masks: Tensor
    cropped_images: Tensor
    cropped_masks: Tensor
    class_ids: Tensor
    class_names: t.List[str]
    attributes_labels: t.Optional[Tensor]
    ids: t.Optional[Tensor]
    instance_ids: Tensor
    instances_tensor: t.Optional[Tensor] = None
    metadata: t.Optional[t.List] = None

    def __post_init__(self):
        assert (
            len(self.cropped_masks.shape) == len(self.cropped_images.shape) == 4
        ), f"The cropped image and mask should be a 4D Tensor, got {self.cropped_images.shape} and {self.cropped_masks.shape}"
        assert (
            len(self.images.shape) == len(self.masks.shape) == 4
        ), f"The image and mask should be a 4D Tensor, got {self.images.shape} and {self.masks.shape}"
        assert (
            self.images.shape[0] == self.masks.shape[0]
        ), f"The batch size is inconsistent, got {self.images.shape} and {self.masks.shape}"
        assert (
            self.images.shape[-2:] == self.masks.shape[-2:]
        ), f"Image and mask H&W inconsistency, got {self.images.shape} and {self.masks.shape}"
        assert (
            self.cropped_images.shape[-2:] == self.cropped_masks.shape[-2:]
        ), f"The size of the cropped mask ({self.cropped_masks.shape}) and cropped image ({self.cropped_images.shape}) should be equal"
        assert (
            self.cropped_masks.shape[1] == 1
        ), f"The size of the channels of the mask should be 1, got {self.cropped_masks.shape}"
        if self.instances_tensor is not None:
            assert self.instances_tensor.shape[1] == 6, self.instances_tensor.shape
            assert len(self.instances_tensor.shape) == 4, self.instances_tensor.shape
        self.class_ids = self.class_ids.to(dtype=torch.int64)
        assertion_list = [
            len(self.class_names),
            len(self.images),
            len(self.masks),
            self.cropped_masks.shape[0],
            self.cropped_images.shape[0],
            self.class_ids.shape[0],
            self.instance_ids.shape[0],
        ]
        if self.attributes_labels is not None:
            assertion_list.append(self.attributes_labels.shape[0])
        if self.ids is not None:
            assertion_list.append(self.ids.shape[0])
        if self.instances_tensor is not None:
            assertion_list.append(self.instances_tensor.shape[0])
        if self.metadata:
            assertion_list.append(len(self.metadata))
        assert (
            len(set(int(x) for x in assertion_list)) == 1
        ), "The batch size should be the same over all elements of the batch"

    @staticmethod
    def collate(model_input_items: t.Sequence[ModelInputItem]) -> "ModelInputItems":
        return ModelInputItems(
            images=torch.stack([item.image for item in model_input_items], dim=0),
            masks=torch.stack([item.mask for item in model_input_items], dim=0),
            cropped_images=torch.stack([item.cropped_image for item in model_input_items], dim=0),
            cropped_masks=torch.stack([item.cropped_mask for item in model_input_items], dim=0),
            class_ids=torch.stack([item.class_id for item in model_input_items], dim=0),
            class_names=[item.class_name for item in model_input_items],
            attributes_labels=torch.stack(
                [item.attributes_label for item in model_input_items], dim=0
            )
            if model_input_items[0].attributes_label is not None
            else None,
            ids=torch.stack([item.id for item in model_input_items], dim=0)
            if model_input_items[0].id is not None
            else None,
            instance_ids=torch.stack([item.instance_id for item in model_input_items], dim=0),
            instances_tensor=torch.stack(
                [item.instances_tensor for item in model_input_items], dim=0
            )
            if model_input_items[0].instances_tensor is not None
            else None,
            metadata=[item.metadata for item in model_input_items],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> ModelInputItem:
        return ModelInputItem(
            image=self.images[idx],
            mask=self.masks[idx],
            cropped_image=self.cropped_images[idx],
            cropped_mask=self.cropped_masks[idx],
            class_id=self.class_ids[idx],
            class_name=self.class_names[idx],
            attributes_label=self.attributes_labels[idx]
            if self.attributes_labels is not None
            else None,
            id=self.ids[idx] if self.ids is not None else None,
            instance_id=self.instance_ids[idx],
            instances_tensor=self.instances_tensor[idx]
            if self.instances_tensor is not None
            else None,
            metadata=self.metadata[idx],
        )

    def __iter__(self) -> ModelInputItem:
        for idx in range(len(self)):
            yield self[idx]

    def to_device(self, device: torch.device):
        return ModelInputItems(
            images=self.images.to(device),
            masks=self.masks.to(device),
            class_names=self.class_names,
            cropped_images=self.cropped_images.to(device),
            cropped_masks=self.cropped_masks.to(device),
            class_ids=self.class_ids.to(device),
            attributes_labels=self.attributes_labels.to(device)
            if self.attributes_labels is not None
            else None,
            ids=self.ids.to(device) if self.ids is not None else None,
            instance_ids=self.instance_ids.to(device),
            instances_tensor=self.instances_tensor.to(device)
            if self.instances_tensor is not None
            else None,
            metadata=self.metadata,
        )

    @property
    def classes_set(self) -> t.Set:
        return set(self.class_ids.cpu().tolist())

    @property
    def from_single_class(self) -> bool:
        return len(self.classes_set) == 1

    @property
    def single_class_id(self) -> int:
        assert self.from_single_class
        return self.classes_set.pop()

    @property
    def single_class_name(self) -> str:
        assert self.from_single_class
        return self.class_names[0]


@config_register("ModelOutput")
class VAWModelOutput(ABCModelOutput):
    PreLogits: Tensor
    G: Tensor
    E: t.List[Tensor]


@config_register("ModelOutput")
class GlidePhase1Output(ABCModelOutput):
    PreLogits: t.Literal[None]
    objects: Tensor
    mask: Tensor
    category: Tensor
    attributes_local: Tensor
    attributes_intrinsic: Tensor

    def __post_init__(self):
        assert self.PreLogits is None
        assert (
            self.mask.shape[1] == 1
        ), f"mask should be a single channel Tensor, got {self.mask.shape}"
        assert len(self.mask.shape) == 4, f"mask should be a 4D Tensor, got {self.mask.shape}"
        assert (
            len(self.category.shape) == 2
        ), f"Category should be 2D Tensor, got {self.category.shape}"
        assert (
            len(self.objects.shape) == 4
        ), f"Objects should be a 4D Tensor, got {self.objects.shape}"
        assert (
            self.objects.shape[1] >= 6
        ), f"Number of channels for objects should be 6, got {self.objects.shape}"
        assert (
            len(self.attributes_local.shape) == 2
        ), f"local attributes should be 2D Tensor, got {self.attributes_local.shape}"
        assert (
            len(self.attributes_intrinsic.shape) == 2
        ), f"intrinsic attributes should be 2D Tensor, got {self.attributes_intrinsic.shape}"


@config_register("ModelOutput")
class GlidePhase2Output(ABCModelOutput):
    PreLogits: Tensor
    category_embedding: t.Optional[Tensor]
    attributes_embedding: Tensor
    category_id: Tensor

    def __post_init__(self):
        self.category_id = torch.tensor(list(set(self.category_id.view(-1).tolist())))
        assert (
            len(self.category_id) == 1
        ), f"We should only have a single category for a single batch, got {self.category_id}"
        # assert (
        #     self.mask.shape[1] == 1
        # ), f"mask should be a single channel Tensor, got {self.mask.shape}"
        # assert len(self.mask.shape) == 4, f"mask should be a 4D Tensor, got {self.mask.shape}"
        if self.category_embedding is not None:
            assert (
                len(self.category_embedding.shape) == 2
            ), f"Category should be 2D Tensor, got {self.category_embedding.shape}"
        assert (
            len(self.attributes_embedding.shape) == 2
        ), f"attributes should be 2D Tensor, got {self.attributes_embedding.shape}"
        assert (
            len(self.PreLogits.shape) == 2
        ), f"attributes should be 2D Tensor, got {self.PreLogits.shape}"


@config_register("ModelConfig")
class GeneralModelConfig(ABCModelConfig):
    backbone: str
    type: str
    fc_dims: t.List[int]
    pretrained_state_dict: t.Optional[str] = None
    intermediate: t.Optional[t.List[int]] = None
    backbone_drop_last: t.Optional[bool] = False  # drop the last FC layer when loading backbone

    def __post_init__(self):
        if not self.backbone_drop_last:
            self.backbone_drop_last = False


@config_register("ModelConfig")
class VAWModelConfig(ABCModelConfig):
    losses: VAWLossesConfig
    type: str = "vaw_paper"
    is_multi_head: bool = False
    backbone: str = "resnet50"
    features_dims: t.Tuple[int, int, int] = (512, 1024, 2048)
    class_gate_dims: t.Tuple[int, int, int] = (100, 128, 2048)  # [input, intermediate, output]
    object_localizer_channels: t.Tuple[int, int, int] = (
        2048,
        256,
        1,
    )  # [input, intermediate, output]
    multi_attention_channels: t.Tuple[int, int, int] = (
        2048,
        256,
        1,
    )  # [input, intermediate, output]
    multi_attention_proj_channels: int = 128
    n_multi_attention: int = 3
    pretrained_state_dict: t.Optional[str] = None
    intermediate: t.Optional[t.List[int]] = None
    backbone_drop_last: t.Optional[bool] = False  # drop the last FC layer when loading backbone
    glove_file_path: str = "s3://scale-ml/home/kareemmetwaly/datasets/glove/glove.6B.100d.txt"

    def __post_init__(self):
        if not self.backbone_drop_last:
            self.backbone_drop_last = False


@config_register("LossConfig")
class GlideLossesConfig(ABCLossConfig):
    weights: t.Mapping[str, float]

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "GlideLossesConfig":
        return deserialize(GlideLossesConfig, d)


@dataclass
class UpsampleConfig:
    type: str
    scale_factor: int


@dataclass
class DownsampleConfig:
    type: str
    kernel: int
    stride: int
    padding: int

    def __post_init__(self):
        assert self.type in ["maxpool", "avgpool"]


@dataclass
class Conv2dConfig:
    in_channels: int
    out_channels: t.Optional[int]
    kernel_size: int = 1
    stride: t.Optional[int] = 1
    dilation: t.Optional[int] = 1
    padding: t.Optional[int] = 0
    groups: t.Optional[int] = 1
    bias: t.Optional[bool] = True
    norm: t.Optional[bool] = True
    activation: t.Optional[str] = None
    upsample: t.Optional[UpsampleConfig] = None
    downsample: t.Optional[DownsampleConfig] = None

    def __post_init__(self):
        self.activation = self.activation.lower() if self.activation else self.activation
        if not self.stride:
            self.stride = 1
        if not self.dilation:
            self.dilation = 1
        if not self.padding:
            self.padding = 0
        if not self.groups:
            self.groups = 1
        if not self.bias:
            self.bias = True
        if not self.norm:
            self.norm = True
        assert (not self.downsample) or (
            not self.upsample
        ), "we cannot have upsampling and downsampling in the same conv layer"

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "Conv2dConfig":
        return deserialize(Conv2dConfig, d)


@dataclass
class FullyConnectedConfig:
    in_channels: t.Optional[int]
    out_channels: t.Optional[int]
    bias: t.Optional[bool] = True
    norm: t.Optional[bool] = False
    activation: t.Optional[str] = None
    dropout: t.Optional[float] = None

    def __post_init__(self):
        self.activation = self.activation.lower() if self.activation else self.activation

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "FullyConnectedConfig":
        return deserialize(FullyConnectedConfig, d)


@dataclass
class DescriptorConfig:
    category_fc: t.Sequence[
        FullyConnectedConfig
    ]  # dimensions of the fully connected network for the category gating
    mask_convs: t.Sequence[
        Conv2dConfig
    ]  # Parameters for the convolution blocks for the mask module
    post_convs: t.Sequence[Conv2dConfig]  #

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "DescriptorConfig":
        return deserialize(DescriptorConfig, d)


@dataclass
class MultiheadConfig:
    n_classes: t.Optional[int]
    fc_config: t.Sequence[FullyConnectedConfig]

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "MultiheadConfig":
        return deserialize(MultiheadConfig, d)


@dataclass
class ProjectionConfig:
    # TODO: create projection configuration

    @staticmethod
    def from_dict(d: t.Mapping[str, t.Any]) -> "ProjectionConfig":
        return deserialize(ProjectionConfig, d)


class InterpreterType(Enum):
    Multihead = "multihead"
    Projection = "projection"


@config_register("ModelConfig")
class GlideModelConfig(ABCModelConfig):
    losses: GlideLossesConfig

    descriptor: DescriptorConfig
    classifier: t.Sequence[FullyConnectedConfig]
    interpreter: t.Union[MultiheadConfig, ProjectionConfig]

    global_decoder: t.Sequence[Conv2dConfig]
    mask_decoder: t.Sequence[Conv2dConfig]
    category_embedding: t.Sequence[FullyConnectedConfig]
    local_attributes_decoder: t.Sequence[FullyConnectedConfig]
    intrinsic_attributes_decoder: t.Sequence[FullyConnectedConfig]

    gate: t.Sequence[Conv2dConfig]
    type: str = "GlideNet"
    backbone: str = "resnet50"
    squeeze: t.Optional[int] = None
    pretrained_state_dict: t.Optional[str] = None

    flags: t.Optional[Flags] = None

    def __post_init__(self):
        if isinstance(self.flags, dict):
            self.flags = Flags(self.flags)
        elif self.flags is None:
            self.flags = Flags()
        assert isinstance(self.flags, Flags), type(self.flags)

    @property
    def interpreter_type(self):
        if isinstance(self.interpreter, MultiheadConfig):
            return InterpreterType.Multihead
        elif isinstance(self.interpreter, ProjectionConfig):
            return InterpreterType.Projection
        else:
            raise NotImplementedError


ModelConfig = t.Union[GlideModelConfig, VAWModelConfig, GeneralModelConfig]
