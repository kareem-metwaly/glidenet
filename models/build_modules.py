import typing as t
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision import models  # noqa: F401

from models import informed_resnet
from structures.model import Conv2dConfig, DescriptorConfig, FullyConnectedConfig


class Squeeze(nn.Module):
    def __init__(self, dims: t.Sequence[int]):
        super(Squeeze, self).__init__()
        self.dims = sorted(dims, reverse=True)

    def forward(self, x: Tensor):
        for dim in self.dims:
            try:
                x = x.squeeze(dim=dim)
            except:
                pass
        return x


class GenericSequential(nn.Sequential):
    """
    Similar to nn.Sequential but deals with flexible size for the input.
    It assumes that the first module will take arbitrary number of inputs, but the following layers will only take the output of the preceding layer.
    """

    def forward(self, x, *args, **kwargs):
        for i, module in enumerate(self):
            x = module(x, *args, **kwargs)
            if isinstance(x, tuple or list):
                if isinstance(x[1], dict):
                    x, kwargs = x
                    args = {}
                else:
                    args = x[1:]
                    x = x[0]
                    kwargs = {}
            else:
                kwargs = {}
                args = {}
        if args:
            return (x, *args)
        if kwargs:
            return x, kwargs
        return x


class SplittedBackbone(nn.Module):
    def __init__(
        self,
        modules: t.Sequence[t.Tuple[str, nn.Module]],
        squeeze: t.Optional[t.OrderedDict[str, nn.Module]] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for name, module in modules:
            self.blocks.update({name: module})

        if squeeze:
            self.squeeze_blocks = nn.ModuleDict()
            for name, module in squeeze.items():
                self.squeeze_blocks.update({name: module})
        else:
            self.squeeze_blocks = None

    def forward(self, x: torch.Tensor, **kwargs) -> t.Sequence[t.Tuple[str, torch.Tensor]]:
        if "return_names" in kwargs:
            return_names = kwargs.pop("return_names")
        else:
            return_names = False
        out = []
        for i, (name, module) in enumerate(self.blocks.items()):
            if i != 0:
                if isinstance(x, tuple or list):
                    x, kwargs = x
            x = module(x, **kwargs)
            if self.squeeze_blocks:
                x_squeezed = (
                    self.squeeze_blocks[f"squeeze_{name}"](x[0])
                    if isinstance(x, tuple or list)
                    else self.squeeze_blocks[f"squeeze_{name}"](x)
                )
                out.append((name, x_squeezed))
            else:
                out.append((name, x))
        return out if return_names else [module for _, module in out]


class Descriptor(nn.Module):
    def __init__(self, configs: DescriptorConfig):
        super(Descriptor, self).__init__()
        self.category_in = build_fc_network(configs.category_fc).Model
        self.mask_in = build_conv2d_network(configs.mask_convs).Model
        self.post = build_conv2d_network(configs.post_convs).Model

    def forward(self, category: Tensor, mask: Tensor) -> Tensor:
        assert len(category.shape) == 2, category.shape
        assert mask.shape[1] == 1, mask.shape
        category = self.category_in(category)
        mask = self.mask_in(mask)
        mask = category.unsqueeze(2).unsqueeze(3) * mask
        mask = self.post(mask)
        return mask


class BuildOutput(t.NamedTuple):
    Model: t.Union[
        nn.Module, Descriptor, Squeeze, SplittedBackbone, GenericSequential, nn.Sequential
    ]
    OutputSize: t.Optional[torch.Size] = None


def sequential2generic(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.Sequential):
        modules = OrderedDict()
        modules.update({k: v for k, v in model.named_children()})
        model = GenericSequential(modules)
    for name, module in model.named_children():
        if any(isinstance(m, nn.Sequential) for m in module.modules()):
            module = sequential2generic(module)
            model.__setattr__(name, module)
    return model


def build_fc_norm_actv_layer(configs: FullyConnectedConfig, test: bool = True) -> BuildOutput:
    fc_bias = (not configs.norm) and configs.bias
    modules = OrderedDict(
        {
            "FC": nn.Linear(
                in_features=configs.in_channels, out_features=configs.out_channels, bias=fc_bias
            ),
        }
    )
    if configs.norm:
        modules.update(
            {"norm": nn.BatchNorm1d(num_features=configs.out_channels, affine=configs.bias)}
        )

    if configs.dropout:
        assert 0 <= configs.dropout <= 1, configs.dropout
        modules.update({"dropout": nn.Dropout(p=configs.dropout)})

    if configs.activation:
        if configs.activation == "relu":
            modules.update({"activation": nn.ReLU(inplace=True)})
        elif configs.activation == "softmax":
            modules.update({"activation": nn.Softmax(dim=1)})
        elif configs.activation == "sigmoid":
            modules.update({"activation": nn.Sigmoid()})
        else:
            raise NotImplementedError(f"{configs.activation} is not implemented")

    model = nn.Sequential(modules)
    if test:
        x = torch.rand([2, configs.in_channels])
        x = model(x)
        assert x.shape[1] == configs.out_channels
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_fc_network(
    configs: t.Sequence[FullyConnectedConfig], pre_squeeze: bool = False, test: bool = True
) -> BuildOutput:
    modules = OrderedDict()
    if pre_squeeze:
        modules.update({"Squeeze": Squeeze(dims=[2, 3])})
    for i, c in enumerate(configs):
        modules.update({f"FC_{i}": build_fc_norm_actv_layer(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0].in_channels])
        if pre_squeeze:
            x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_conv2d(configs: Conv2dConfig, test: bool = True) -> BuildOutput:
    conv_bias = (not configs.norm) and configs.bias
    modules = OrderedDict(
        {
            "conv": nn.Conv2d(
                in_channels=configs.in_channels,
                out_channels=configs.out_channels,
                kernel_size=configs.kernel_size,
                stride=configs.stride,
                padding=configs.padding,
                dilation=configs.dilation,
                bias=conv_bias,
            ),
        }
    )
    if configs.norm:
        modules.update(
            {"norm": nn.BatchNorm2d(num_features=configs.out_channels, affine=configs.bias)}
        )

    if configs.activation:
        if configs.activation == "relu":
            modules.update({"activation": nn.ReLU(inplace=True)})
        elif configs.activation == "softmax2d":
            modules.update({"activation": nn.Softmax2d()})
        elif configs.activation == "sigmoid":
            modules.update({"activation": nn.Sigmoid()})
        else:
            raise NotImplementedError(f"{configs.activation} is not implemented")

    if configs.upsample:
        modules.update(
            {
                "upsample": nn.Upsample(
                    scale_factor=configs.upsample.scale_factor, mode=configs.upsample.type
                )
            }
        )

    if configs.downsample:
        if configs.downsample.type == "maxpool":
            modules.update(
                {
                    "downsample": nn.MaxPool2d(
                        kernel_size=configs.downsample.kernel,
                        stride=configs.downsample.stride,
                        padding=configs.downsample.padding,
                    )
                }
            )
        else:
            raise NotImplementedError

    model = nn.Sequential(modules)
    if test:
        x = torch.rand([2, configs.in_channels, 8, 8])
        x = model(x)
        assert x.shape[1] == configs.out_channels
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_conv2d_network(configs: t.Sequence[Conv2dConfig], test: bool = True) -> BuildOutput:
    modules = OrderedDict()
    for i, c in enumerate(configs):
        modules.update({f"Conv_{i}": build_conv2d(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0].in_channels, 224, 224])
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_descriptor(configs: DescriptorConfig, test: bool = True) -> BuildOutput:
    model = Descriptor(configs)
    if test:
        cat = torch.rand([2, configs.category_fc[0].in_channels])
        mask = torch.rand([2, configs.mask_convs[0].in_channels, 224, 224])
        output = model.forward(category=cat, mask=mask)
        assert (
            output.shape[1] == configs.post_convs[-1].out_channels
        ), f"{output.shape} and {configs.post_convs[-1].out_channels}"
        return BuildOutput(model, output.shape)
    return BuildOutput(model, None)


def build_backbone(model_name: str, drop_last: bool) -> BuildOutput:
    try:
        model = eval(f"models.{model_name}(pretrained=True)")
    except ModuleNotFoundError:
        raise NotImplementedError(f"This backbone is not implemented yet; {model_name}")

    if drop_last:
        chosen_layers = OrderedDict()
        for name, child in model.named_children():
            if name != "fc":
                chosen_layers.update({name: child})
        model = nn.Sequential(chosen_layers)
    return BuildOutput(model, None)


def last_out_channels(module: nn.Module, prev_value: int = 0) -> int:
    """recursively go through children modules to figure out the dimensions of the output channels"""
    if hasattr(module, "out_channels"):
        return module.out_channels
    if len(list(module.named_children())):
        for m in module.children():
            prev_value = last_out_channels(m, prev_value)
        return prev_value
    return prev_value


def split_backbone_levels(
    backbone,
    prefix: str = "backbone",
    keywords: t.Sequence[t.Tuple[str, str]] = (
        ("layer2", "layer2"),
        ("layer3", "layer3"),
        ("layer4", "layer4"),
    ),
    squeeze: t.Optional[t.Union[int, t.Sequence[int]]] = None,
) -> "SplittedBackbone":
    modules = []
    if squeeze:
        squeeze_configs = Conv2dConfig(
            in_channels=0,  # will be set later
            out_channels=0,  # will be set later
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            groups=1,
            bias=True,
            norm=True,
            activation="relu",
        )
        squeeze_dict = OrderedDict()
    else:
        squeeze_configs = None
        squeeze_dict = None
    backbone_iter = backbone.named_children()
    out_channels = 0
    for i, (key, stop) in enumerate(keywords):
        ordered_dict = OrderedDict()
        for name, module in backbone_iter:
            ordered_dict.update({name: module})
            out_channels = last_out_channels(module, out_channels)
            if name == stop:
                break
        if squeeze_configs:
            squeeze_configs.in_channels = out_channels
            squeeze_configs.out_channels = squeeze if isinstance(squeeze, int) else squeeze[i]
            squeeze_dict.update({f"squeeze_{prefix}_{key}": build_conv2d(squeeze_configs).Model})
        modules.append((f"{prefix}_{key}", sequential2generic(nn.Sequential(ordered_dict))))

    return SplittedBackbone(modules, squeeze=squeeze_dict)


def build_fc(
    fc_dims: t.Sequence[int],
    intermediate_activations: t.Union[str, t.Sequence[str]] = "ReLU",
) -> nn.Module:
    if isinstance(intermediate_activations, str):
        eval_str = (
            "nn.ReLU(inplace=True)"
            if intermediate_activations == "ReLU"
            else f"nn.{intermediate_activations}()"
        )
        intermediate_activations = [eval(eval_str) for _ in fc_dims[:-1]]
    else:
        assert (
            len(fc_dims) == len(intermediate_activations) + 1
        ), f"{len(fc_dims)} != {len(intermediate_activations)} + 1"
        intermediate_activations = [
            eval(f"nn.{a}(inplace=True)" if a == "ReLU" else f"nn.{a}()")
            for a in intermediate_activations
        ]

    modules = [nn.Linear(fc_dims[0], fc_dims[1])]
    if len(fc_dims) > 2:
        for inp, out, actv in zip(fc_dims[1:-1], fc_dims[2:], intermediate_activations):
            modules.append(actv)
            modules.append(nn.Linear(inp, out))
    return nn.Sequential(*modules)


def build_cnn(
    cnn_dims: t.Sequence[int],
    kernels: t.Sequence[int],
    strides: t.Sequence[int],
    paddings: t.Sequence[int],
    intermediate_activations: t.Union[str, t.Sequence[str]] = "ReLU",
    bias: bool = True,
) -> nn.Module:
    assert len(kernels) == len(strides)
    assert len(strides) == len(paddings)
    assert len(paddings) == len(cnn_dims) - 1
    if not isinstance(intermediate_activations, str):
        assert (
            len(cnn_dims) == len(intermediate_activations) + 1
        ), f"{len(cnn_dims)} != {len(intermediate_activations)} + 1"
        intermediate_activations = [
            eval(f"nn.{a}(inplace=True)" if a == "ReLU" else f"nn.{a}()")
            for a in intermediate_activations
        ]
    else:
        eval_str = (
            "nn.ReLU(inplace=True)"
            if intermediate_activations == "ReLU"
            else f"nn.{intermediate_activations}()"
        )
        intermediate_activations = [eval(eval_str) for _ in cnn_dims[:-1]]
    modules = [
        nn.Conv2d(
            cnn_dims[0],
            cnn_dims[1],
            kernel_size=kernels[0],
            stride=strides[0],
            padding=paddings[0],
            bias=bias,
        )
    ]
    if len(cnn_dims) > 2:
        for inp, out, k, s, p, actv in zip(
            cnn_dims[1:-1],
            cnn_dims[2:],
            kernels[1:],
            strides[1:],
            paddings[1:],
            intermediate_activations,
        ):
            modules.append(actv)
            modules.append(nn.Conv2d(inp, out, kernel_size=k, stride=s, padding=p, bias=bias))
    return nn.Sequential(*modules)


def build_informed_backbone(model_name: str, drop_last: bool) -> informed_resnet.InformedConv2d:
    try:
        model = eval(f"informed_resnet.informed{model_name}(pretrained=True)")
    except ModuleNotFoundError:
        raise NotImplementedError(f"This backbone is not implemented yet; {model_name}")

    if drop_last:
        if hasattr(model, "features"):
            model.forward = model.features
        else:
            raise NotImplementedError(
                "Since the input is an image + mask, using sequential approach wouldn't be recommended"
            )
    return model


class DenseBlock(nn.Module):
    def __init__(self, configs: t.Sequence[Conv2dConfig]):
        super(DenseBlock, self).__init__()
        modules = []
        for i, c in enumerate(configs):
            modules.append(build_conv2d(c).Model)

        self.modules = nn.ModuleList(modules)

    def forward(self, x: Tensor) -> Tensor:
        out = []
        for layer in self.modules:
            x = layer(x)
            out.append(x)
        return torch.cat(out, dim=1)


def build_dense_block(configs: t.Sequence[Conv2dConfig], test: bool = True) -> BuildOutput:
    model = DenseBlock(configs)

    if test:
        x = torch.rand([2, configs[0].in_channels, 224, 224])
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)


def build_dense_network(
    configs: t.Sequence[t.Sequence[Conv2dConfig]], test: bool = True
) -> BuildOutput:
    modules = OrderedDict()
    for i, c in enumerate(configs):
        if i % 2:
            modules.update({f"Dense_{i}": build_dense_block(c).Model})
        else:
            modules.update({f"Transition_{i}": build_conv2d_network(c).Model})
    model = nn.Sequential(modules)

    if test:
        x = torch.rand([2, configs[0][0].in_channels, 224, 224])
        x = model(x)
        return BuildOutput(model, x.shape)
    return BuildOutput(model, None)
