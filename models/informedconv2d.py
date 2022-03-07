import typing as t

import torch
from torch import Tensor, nn


class InformedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if "normalize" in kwargs:
            self.normalize = kwargs["normalize"]
        else:
            self.normalize = True
        if "epsilon" in kwargs:
            self.epsilon = torch.tensor(kwargs["epsilon"])
        else:
            self.epsilon = torch.tensor(1e-6)

        super().__init__(*args, **kwargs)
        self.maskUpdater = nn.Conv2d(
            1,
            1,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=1,
            bias=False,
        )
        self.maskUpdater.weight = nn.Parameter(
            torch.ones((1, 1, self.kernel_size[0], self.kernel_size[1]))
            / (self.kernel_size[0] * self.kernel_size[1]),
            requires_grad=False,
        )

    def forward(self, input: Tensor, mask: Tensor) -> t.Tuple[Tensor, Tensor]:
        updated_mask = self.maskUpdater(mask)
        if self.normalize:
            updated_mask = updated_mask / torch.max(updated_mask.max(), self.epsilon)

        raw_out = super().forward(torch.mul(input, mask))
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (
                torch.div(raw_out - bias_view, torch.max(updated_mask, self.epsilon)) + bias_view
            )
        else:
            output = torch.div(raw_out, torch.max(updated_mask, self.epsilon))

        return output, updated_mask


if __name__ == "__main__":
    inp = torch.rand([2, 3, 224, 224])
    mask = torch.randint(0, 2, [2, 1, 224, 224]).to(torch.float)
    model = InformedConv2d(in_channels=3, out_channels=5, kernel_size=5, padding=2)
    out, mask_out = model(inp, mask)
