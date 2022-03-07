import typing as t
from abc import ABC, abstractmethod

import torch
from torch import nn

from models.build_modules import build_backbone, build_fc
from models.losses import calc_losses
from structures.abstract_classes import ABCModelOutput
from structures.loss import GeneralLoss
from structures.model import ModelConfig, ModelInputItems


class Base(nn.Module, ABC):
    def __init__(self, configs: ModelConfig, classes: t.List[str], attributes: t.List[str]):
        super().__init__()
        self.configs = configs

        self.backbone = build_backbone(
            model_name=self.configs.backbone, drop_last=self.configs.backbone_drop_last
        ).Model
        if self.configs.intermediate:
            self.intermediate = build_fc(self.configs.intermediate)
        self.head = nn.ModuleList(
            [build_fc(self.configs.fc_dims + [len(attributes)]) for _ in classes]
        )
        self.loss_fn = calc_losses

    def forward(
        self, model_items: ModelInputItems, phase_number: t.Optional[int] = None
    ) -> GeneralLoss:
        if phase_number:
            output = self.__getattribute__(f"layers_phase{phase_number}")(model_items)
        else:
            output = self.layers(model_items)
        return self.loss_fn(output, model_items)

    @abstractmethod
    def inference(self, model_items: ModelInputItems) -> torch.Tensor:
        output = self.layers(model_items).PreLogits
        return output

    def layers(self, model_items: ModelInputItems) -> ABCModelOutput:
        head_ids = set(idx.item() for idx in model_items.class_ids)
        assert len(head_ids) == 1, f"the number of classes/batch should be only 1; {head_ids}"
        output = self.head[head_ids.pop()](self.features(model_items))
        return ABCModelOutput(PreLogits=output)

    def features(self, model_items: ModelInputItems) -> torch.Tensor:
        images = model_items.cropped_images
        output = self.backbone(images).squeeze(dim=3).squeeze(dim=2)
        if hasattr(self, "intermediate"):
            output = self.intermediate(output)
        return output

    def on_mode_epoch_end(
        self, mode: str, cache: t.Dict[str, t.Any], classes: t.Set
    ) -> t.Dict[str, float]:
        # sum_vals = {k: sum(v) for k, v in cache.items()}
        # outputs, gts = torch.cat(cache["output"], dim=0), torch.cat(cache["gt"], dim=0)
        # mAP = vaw_utils.calculate_mean_average_precision(outputs, gts)
        # PC_Recall = sum(
        #     sum_vals[f"TP_{cls}"] / max(sum_vals[f"P_{cls}"], 1e-5) for cls in classes
        # ) / len(classes)
        # OV_Recall = sum(sum_vals[f"TP_{cls}"] for cls in classes) / sum(
        #     sum_vals[f"P_{cls}"] for cls in classes
        # )
        # OV_Precision = sum(sum_vals[f"TP_{cls}"] for cls in classes) / sum(
        #     sum_vals[f"PP_{cls}"] for cls in classes
        # )
        # OV_F1 = 2 * OV_Recall * OV_Precision / max(OV_Recall + OV_Precision, 1e-5)
        # mA = (
        #         0.5
        #         * sum(
        #     sum_vals[f"TP_{cls}"] / max(sum_vals[f"P_{cls}"], 1e-5)
        #     + sum_vals[f"TN_{cls}"] / max(sum_vals[f"N_{cls}"], 1e-5)
        #     for cls in classes
        # )
        #         / len(classes)
        # )
        # records = {
        #     "F1": OV_F1,
        #     "mR": PC_Recall,
        #     "mA": mA,
        #     "mAP": mAP,
        # }
        # records = {f"{mode}/{k}": v.item() if hasattr(v, "item") else v for k, v in records.items()}
        # return records
        raise NotImplementedError


if __name__ == "__main__":
    model = Base(
        ModelConfig.from_dict(
            {
                "type": "resnet101",
                "fc_dims": [100, 23, 10],
            }
        ),
        ["cat", "dog"],
        ["standing", "white"],
    )
