import tempfile

import sh
import torch
import torch.nn as nn

from models.informedconv2d import InformedConv2d

# import torch.utils.model_zoo as model_zoo


__all__ = [
    "InformedResNet",
    "informedresnet18",
    "informedresnet34",
    "informedresnet50",
    "informedresnet101",
    "informedresnet152",
]

model_s3 = {
    "informedresnet18": "",
    "informedresnet34": "",
    "informedresnet50": "",
    "informedresnet101": "",
    "informedresnet152": "",
}


def load_model_state_dict_from_s3(s3_location):
    with tempfile.NamedTemporaryFile() as tempf:
        sh.aws.s3.cp(s3_location, tempf.name)
        with open(tempf.name, "rb") as f:
            pretrained_state_dict = torch.load(f)
    return pretrained_state_dict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 informed convolution"""
    return InformedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask):
        residual = x

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, {"mask": mask}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = InformedConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = InformedConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = InformedConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask):
        residual = x
        res_mask = mask

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, mask = self.conv3(out, mask)
        out = self.bn3(out)

        if self.downsample is not None:
            residual, res_mask = self.downsample(x, mask=res_mask)

        out += residual
        out = self.relu(out)

        return out, {"mask": mask}


class InformedBatchNorm2d(nn.BatchNorm2d):
    """
    Bypasses the mask in normalization process
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().forward(input), mask


class CBRM(nn.Module):
    def __init__(self):
        super(CBRM, self).__init__()
        self.conv1 = InformedConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        mask = self.maxpool(mask)
        return x, {"mask": mask}


class InformedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super().__init__()
        self.cbrm = CBRM()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, InformedConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                InformedConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                InformedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x, mask):
        x, mask = self.cbrm(x, mask)
        x, mask = self.layer1(x, mask)
        x, mask = self.layer2(x, mask)
        x, mask = self.layer3(x, mask)
        x, mask = self.layer4(x, mask)
        x = self.avgpool(x)
        return x

    def forward(self, x, mask):
        x = self.features(x, mask)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def informedresnet18(pretrained=False, **kwargs):
    """Constructs an InformedResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InformedResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_model_state_dict_from_s3(model_s3["informedresnet18"]))
    return model


def informedresnet34(pretrained=False, **kwargs):
    """Constructs an InformedResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InformedResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_model_state_dict_from_s3(model_s3["informedresnet34"]))
    return model


def informedresnet50(pretrained=False, **kwargs):
    """Constructs a InformedResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InformedResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_model_state_dict_from_s3(model_s3["informedresnet50"]))
    return model


def informedresnet101(pretrained=False, **kwargs):
    """Constructs a InformedResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InformedResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_model_state_dict_from_s3(model_s3["informedresnet101"]))
    return model


def informedresnet152(pretrained=False, **kwargs):
    """Constructs a InformedResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InformedResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_model_state_dict_from_s3(model_s3["informedresnet152"]))
    return model
