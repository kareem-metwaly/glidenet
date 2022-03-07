import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.informed_resnet import Bottleneck, InformedResNet


pdresnet50_url = "https://www.dropbox.com/s/yauo6kosqo572we/model_best.pth.tar?dl=1"
save_location = "/home/krm/Downloads/informed_resnet50.pth"


def generate_param(kernel_size):
    return nn.Parameter(
        torch.ones((1, 1, kernel_size[0], kernel_size[1])) / (kernel_size[0] * kernel_size[1]),
        requires_grad=False,
    )


model = InformedResNet(Bottleneck, [3, 4, 6, 3])
state_dict = {
    k.split("module.")[-1]: v for k, v in model_zoo.load_url(pdresnet50_url)["state_dict"].items()
}

keys_equivalence = [
    ("cbrm.conv1.weight", "conv1.weight"),
    ("cbrm.bn1.weight", "bn1.weight"),
    ("cbrm.bn1.bias", "bn1.bias"),
    ("cbrm.bn1.running_mean", "bn1.running_mean"),
    ("cbrm.bn1.running_var", "bn1.running_var"),
]

missing_keys = [
    "cbrm.conv1.maskUpdater.weight",
    "layer1.0.conv1.maskUpdater.weight",
    "layer1.0.conv2.maskUpdater.weight",
    "layer1.0.conv3.maskUpdater.weight",
    "layer1.0.downsample.0.maskUpdater.weight",
    "layer1.1.conv1.maskUpdater.weight",
    "layer1.1.conv2.maskUpdater.weight",
    "layer1.1.conv3.maskUpdater.weight",
    "layer1.2.conv1.maskUpdater.weight",
    "layer1.2.conv2.maskUpdater.weight",
    "layer1.2.conv3.maskUpdater.weight",
    "layer2.0.conv1.maskUpdater.weight",
    "layer2.0.conv2.maskUpdater.weight",
    "layer2.0.conv3.maskUpdater.weight",
    "layer2.0.downsample.0.maskUpdater.weight",
    "layer2.1.conv1.maskUpdater.weight",
    "layer2.1.conv2.maskUpdater.weight",
    "layer2.1.conv3.maskUpdater.weight",
    "layer2.2.conv1.maskUpdater.weight",
    "layer2.2.conv2.maskUpdater.weight",
    "layer2.2.conv3.maskUpdater.weight",
    "layer2.3.conv1.maskUpdater.weight",
    "layer2.3.conv2.maskUpdater.weight",
    "layer2.3.conv3.maskUpdater.weight",
    "layer3.0.conv1.maskUpdater.weight",
    "layer3.0.conv2.maskUpdater.weight",
    "layer3.0.conv3.maskUpdater.weight",
    "layer3.0.downsample.0.maskUpdater.weight",
    "layer3.1.conv1.maskUpdater.weight",
    "layer3.1.conv2.maskUpdater.weight",
    "layer3.1.conv3.maskUpdater.weight",
    "layer3.2.conv1.maskUpdater.weight",
    "layer3.2.conv2.maskUpdater.weight",
    "layer3.2.conv3.maskUpdater.weight",
    "layer3.3.conv1.maskUpdater.weight",
    "layer3.3.conv2.maskUpdater.weight",
    "layer3.3.conv3.maskUpdater.weight",
    "layer3.4.conv1.maskUpdater.weight",
    "layer3.4.conv2.maskUpdater.weight",
    "layer3.4.conv3.maskUpdater.weight",
    "layer3.5.conv1.maskUpdater.weight",
    "layer3.5.conv2.maskUpdater.weight",
    "layer3.5.conv3.maskUpdater.weight",
    "layer4.0.conv1.maskUpdater.weight",
    "layer4.0.conv2.maskUpdater.weight",
    "layer4.0.conv3.maskUpdater.weight",
    "layer4.0.downsample.0.maskUpdater.weight",
    "layer4.1.conv1.maskUpdater.weight",
    "layer4.1.conv2.maskUpdater.weight",
    "layer4.1.conv3.maskUpdater.weight",
    "layer4.2.conv1.maskUpdater.weight",
    "layer4.2.conv2.maskUpdater.weight",
    "layer4.2.conv3.maskUpdater.weight",
]

# treat_missing_as = torch.nn.Parameter(torch.Tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))
model_params = list(model.named_parameters())

for key in missing_keys:
    equivalent_conv = key.replace("maskUpdater.", "")
    param = [param for name, param in model_params if name == equivalent_conv]
    assert len(param) == 1
    param = param[0]
    kernel_size = param.shape[-2:]
    state_dict.update({key: generate_param(kernel_size)})

for key_missing, key_found in keys_equivalence:
    state_dict.update({key_missing: state_dict.pop(key_found)})

# to test
model.load_state_dict(state_dict)


# write state_dict
torch.save(state_dict, save_location)
print("DONE!")
