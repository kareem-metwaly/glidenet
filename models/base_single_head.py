import typing as t

import PIL.Image
import torch

from models.base import Base
from models.build_modules import build_backbone, build_fc
from structures.model import ModelConfig


class Base_single_head(Base):
    def __init__(self, configs: ModelConfig, classes: t.List[str], attributes: t.List[str]):
        super(Base_single_head, self).__init__(
            configs=configs, classes=classes, attributes=attributes
        )
        self.configs = configs

        self.backbone = build_backbone(
            model_name=self.configs.backbone, drop_last=self.configs.backbone_drop_last
        )
        if self.configs.intermediate:
            self.intermediate = build_fc(self.configs.intermediate)
        self.head = build_fc(self.configs.fc_dims + [len(attributes)])

    def layers(
        self, model_item_dicts: t.Dict[str, t.Union[torch.Tensor, PIL.Image.Image]]
    ) -> t.Mapping[str, torch.Tensor]:
        return {"pred_prelogits": self.head(self.features(model_item_dicts))}


if __name__ == "__main__":
    model = Base_single_head(
        ModelConfig.from_dict(
            {
                "type": "resnet101",
                "fc_dims": [100, 23, 10],
            }
        ),
        ["cat", "dog"],
        ["standing", "white"],
    )
