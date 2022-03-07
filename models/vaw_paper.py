import functools
import typing as t

import torch
import torch.nn.functional as F
from torch import nn

from models.base import Base
from models.build_modules import build_cnn, build_fc, split_backbone_levels
from models.glove_word_embedding import GloVe
from structures.loss import GeneralLoss, LossItem
from structures.model import (
    ModelConfig,
    ModelInputItems,
    VAWLossesConfig,
    VAWModelConfig,
    VAWModelOutput,
)


class Vaw_paper(Base):
    def __init__(self, configs: VAWModelConfig, classes: t.List[str], attributes: t.List[str]):
        assert configs.backbone_drop_last, "You must skip the fc layer of the backbone ResNet50"
        assert configs.backbone == "resnet50", configs.backbone
        assert not configs.intermediate, "There shouldn't be an intermediate layer in VAW paper"
        assert configs.features_dims[2] == configs.class_gate_dims[2]
        assert configs.features_dims[2] == configs.object_localizer_channels[0]
        assert configs.features_dims[2] == configs.multi_attention_channels[0]
        assert configs.object_localizer_channels[2] == 1
        assert configs.multi_attention_channels[2] == 1
        parent_configs = ModelConfig(
            backbone=configs.backbone,
            type="VAWPaper",
            fc_dims=[10],
            pretrained_state_dict=configs.pretrained_state_dict,
            intermediate=None,
            backbone_drop_last=True,
        )
        super(Vaw_paper, self).__init__(
            configs=parent_configs, classes=classes, attributes=attributes
        )
        self.configs = configs
        self.loss_fn = functools.partial(vaw_paper_losses, weights=self.configs.losses)

        self.splitted_backbone = split_backbone_levels(
            self.backbone,
            "backbone",
            [("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4")],
        )

        self.embedding = GloVe(
            file_path=self.configs.glove_file_path, size=self.configs.class_gate_dims[0]
        )
        self.gate = nn.Sequential(build_fc(self.configs.class_gate_dims), nn.Sigmoid())
        self.f_rel = build_cnn(
            self.configs.object_localizer_channels,
            kernels=[1 for _ in range(len(self.configs.object_localizer_channels) - 1)],
            strides=[1 for _ in range(len(self.configs.object_localizer_channels) - 1)],
            paddings=[0 for _ in range(len(self.configs.object_localizer_channels) - 1)],
        )
        self.f_att = nn.ModuleList(
            [
                build_cnn(
                    self.configs.multi_attention_channels,
                    kernels=[1 for _ in range(len(self.configs.multi_attention_channels) - 1)],
                    strides=[1 for _ in range(len(self.configs.multi_attention_channels) - 1)],
                    paddings=[0 for _ in range(len(self.configs.multi_attention_channels) - 1)],
                )
                for _ in range(self.configs.n_multi_attention)
            ]
        )
        self.f_proj = nn.ModuleList(
            [
                nn.Linear(
                    self.configs.object_localizer_channels[0],
                    self.configs.multi_attention_proj_channels,
                )
                for _ in range(self.configs.n_multi_attention)
            ]
        )
        self.softmax_2d = nn.Softmax2d()
        z_low_dim = self.configs.features_dims[0] + self.configs.features_dims[1]
        z_rel_dim = self.configs.features_dims[2]
        z_att_dim = self.configs.n_multi_attention * self.configs.multi_attention_proj_channels
        fc_in_size = z_low_dim + z_rel_dim + z_att_dim
        if self.configs.is_multi_head:
            self.n_attributes = {}
            for cls in classes:
                total = sum(1 for s in attributes if s.split("::")[0] == cls)
                if total > 0:
                    self.n_attributes[cls] = total
            heads = {}
            for cls, value in self.n_attributes.items():
                heads.update({cls: build_fc([fc_in_size, value])})
            self.head = nn.ModuleDict(heads)
        else:
            self.head = build_fc([fc_in_size, len(attributes)])
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

    def layers(self, model_items: ModelInputItems) -> VAWModelOutput:
        f1, f2, f3 = self.splitted_backbone(model_items.cropped_images)
        x = self.gate(self.embedding(model_items.class_names)).unsqueeze(2).unsqueeze(2) * f3
        G = self.softmax_2d(self.f_rel(x))
        E = [f_att(x) for f_att in self.f_att]
        z_rel = self.adaptive_avg_pool(G * x).squeeze(3).squeeze(2)
        z_att = torch.cat(
            [
                f_proj(
                    self.adaptive_avg_pool(
                        self.softmax_2d(
                            Ei,
                        )
                        * x
                    )
                    .squeeze(3)
                    .squeeze(2)
                )
                for Ei, f_proj in zip(E, self.f_proj)
            ],
            dim=1,
        )
        z_low = torch.cat(
            [self.adaptive_avg_pool(f).squeeze(3).squeeze(2) for f in [f1, f2]], dim=1
        )
        if self.configs.is_multi_head:
            prelogits = self.head[model_items.single_class_name](
                torch.cat((z_att, z_low, z_rel), dim=1)
            )
        else:
            prelogits = self.head(torch.cat((z_att, z_low, z_rel), dim=1))
        return VAWModelOutput(PreLogits=prelogits, G=G, E=E)

    def on_mode_epoch_end(
        self, mode: str, cache: t.Dict[str, t.Any], classes: t.Set
    ) -> t.Dict[str, float]:
        sum_vals = {k: sum(v) for k, v in cache.items()}
        # gts = torch.cat(cache["gt"], dim=0)
        # outputs = torch.cat(cache["output"], dim=0)
        # mAP = vaw_utils.calculate_mean_average_precision(outputs, gts)
        PC_Recall = sum(
            sum_vals[f"TP_{cls}"] / max(sum_vals[f"P_{cls}"], 1e-5) for cls in classes
        ) / len(classes)
        OV_Recall = sum(sum_vals[f"TP_{cls}"] for cls in classes) / sum(
            sum_vals[f"P_{cls}"] for cls in classes
        )
        OV_Precision = sum(sum_vals[f"TP_{cls}"] for cls in classes) / sum(
            sum_vals[f"PP_{cls}"] for cls in classes
        )
        OV_F1 = 2 * OV_Recall * OV_Precision / max(OV_Recall + OV_Precision, 1e-5)
        mA = (
            0.5
            * sum(
                sum_vals[f"TP_{cls}"] / max(sum_vals[f"P_{cls}"], 1e-5)
                + sum_vals[f"TN_{cls}"] / max(sum_vals[f"N_{cls}"], 1e-5)
                for cls in classes
            )
            / len(classes)
        )
        records = {
            "F1": OV_F1,
            "mR": PC_Recall,
            "mA": mA,
            # "mAP": mAP,
        }
        records = {f"{mode}/{k}": v.item() if hasattr(v, "item") else v for k, v in records.items()}
        return records


def vaw_paper_losses(
    output: VAWModelOutput, model_items: ModelInputItems, weights: VAWLossesConfig
) -> GeneralLoss:
    output_loss = GeneralLoss()

    lambda_div = weights.lambda_div
    lambda_rel = weights.lambda_rel

    w = weights.w  # noqa: F841
    G = output.G
    E = output.E
    M = model_items.cropped_masks
    gt_attributes = model_items.attributes_labels.to(dtype=torch.float32)
    pred_prelogits = output.PreLogits

    P = gt_attributes == 1  # Positive
    N = gt_attributes == -1  # Negative
    U = gt_attributes == 0  # Unlabeled
    assert P.sum() + N.sum() + U.sum() == gt_attributes.numel()

    pos_output = torch.where(P, pred_prelogits, torch.ones_like(pred_prelogits).fill_(-1e10))
    neg_output = torch.where(N, -pred_prelogits, torch.ones_like(pred_prelogits).fill_(-1e10))

    div_loss = 0
    norms_E = {}
    for i in range(len(E)):
        norms_E[i] = E[i].view(-1).dot(E[i].view(-1)).sqrt()
        for j in range(i, len(E)):
            if j not in norms_E:
                norms_E[j] = E[j].view(-1).dot(E[j].view(-1)).sqrt()
            div_loss = div_loss + (E[i].view(-1).dot(E[j].view(-1)) / (norms_E[i] * norms_E[j]))

    M_avg = torch.nn.functional.adaptive_avg_pool2d(M, G.shape[-2:])

    # Calculating losses for backward propagation
    output_loss.add(
        LossItem(
            name="BCE_positive",
            value=F.binary_cross_entropy_with_logits(
                pos_output, P.to(dtype=torch.float32), reduction="mean"
            ),
            isLogged=True,
            isCached=False,
            isBackpropagated=True,
            weight=1.0,
        )
    )
    output_loss.add(
        LossItem(
            name="BCE_negative",
            value=F.binary_cross_entropy_with_logits(
                neg_output, N.to(dtype=torch.float32), reduction="mean"
            ),
            isLogged=True,
            isCached=False,
            isBackpropagated=True,
            weight=1.0,
        )
    )
    output_loss.add(
        LossItem(
            name="rel",
            value=torch.sum(G * (1 - M_avg) - lambda_rel * G * M_avg).squeeze(),
            isLogged=True,
            isCached=False,
            isBackpropagated=True,
            weight=1.0,
        )
    )
    output_loss.add(
        LossItem(
            name="div",
            value=div_loss,
            isLogged=True,
            isCached=False,
            isBackpropagated=True,
            weight=lambda_div,
        )
    )

    # calculating some metrics
    Labeled = P.bitwise_or(N)
    TN = output.PreLogits[N] <= 0.0  # True Negative
    TP = output.PreLogits[P] > 0.0
    FN = output.PreLogits[P] <= 0.0
    FP = output.PreLogits[N] > 0.0  # False Positive
    assert TP.sum() + FN.sum() == P.sum(), "some positive labels are skipped"
    assert TN.sum() + FP.sum() == N.sum(), "some negative labels are skipped"
    PP = output.PreLogits[Labeled] > 0.0  # Predicted Positive (from labeled data P+N only)
    PN = output.PreLogits[Labeled] <= 0.0  # Predicted Negative (from labeled data P+N only)
    PPU = output.PreLogits[U] > 0.0  # unlabeled predicted positive
    PNU = output.PreLogits[U] <= 0.0  # unlabeled predicted negative

    metrics_dict = {
        "accuracy": (TP.sum() + TN.sum()) / (P.sum() + N.sum()) if (P.sum() + N.sum()) > 0 else 1,
        "recall": TP.sum() / P.sum() if P.sum() > 0 else 1,
        "precision": TP.sum() / PP.sum() if PP.sum() > 0 else 1,
        "unlabeled_predicted_positive_rate": PPU.sum() / U.sum() if U.sum() > 0 else 0,
        "unlabeled_predicted_negative_rate": PNU.sum() / U.sum() if U.sum() > 0 else 0,
        "TN": TN.sum(),
        "TP": TP.sum(),
        "FN": FN.sum(),
        "FP": FP.sum(),
        "PP": PP.sum(),
        "PN": PN.sum(),
        "PPUnlabeled": PPU.sum(),
        "PNUnlabeled": PNU.sum(),
    }
    metrics_dict["F1-score"] = (
        2
        * (metrics_dict["recall"] * metrics_dict["precision"])
        / max(metrics_dict["recall"] + metrics_dict["precision"], 1e-4)
    )
    TNR = (TN.sum() / N.sum()) if N.sum() > 0 else 1
    metrics_dict["balanced_accuracy"] = 0.5 * (metrics_dict["recall"] + TNR)
    for k, v in metrics_dict.items():
        output_loss.add(
            LossItem(
                name=k,
                value=v,
                isLogged=True,
                isCached=False,
                isBackpropagated=False,
            )
        )

    # values to be used for average metrics
    assert model_items.from_single_class, model_items.class_ids
    class_id = set(model_items.class_ids).pop().item()
    extra_dict = {
        f"TP_{class_id}": TP.sum(),
        f"TN_{class_id}": TN.sum(),
        f"P_{class_id}": P.sum(),
        f"N_{class_id}": N.sum(),
        f"PP_{class_id}": PP.sum(),
        f"PN_{class_id}": PN.sum(),
        # "output": output.PreLogits,
        # "gt": gt_attributes,
    }
    for k, v in extra_dict.items():
        output_loss.add(
            LossItem(
                name=k,
                value=v,
                isLogged=False,
                isCached=True,
                isBackpropagated=False,
            )
        )
    return output_loss
