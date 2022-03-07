import typing as t

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn.functional import interpolate, softmax
from torchvision.transforms import ToPILImage, ToTensor

from models.base import Base
from models.build_modules import (
    build_backbone,
    build_conv2d_network,
    build_descriptor,
    build_fc_network,
    build_informed_backbone,
    split_backbone_levels,
)
from structures.loss import GeneralLoss, LossItem
from structures.model import (
    GeneralModelConfig,
    GlideLossesConfig,
    GlideModelConfig,
    GlidePhase1Output,
    GlidePhase2Output,
    InterpreterType,
    ModelInputItems,
)


class Glidenet(Base):
    classes: t.List[str]

    def __init__(self, configs: GlideModelConfig, classes: t.List[str], attributes: t.List[str]):
        parent_configs = GeneralModelConfig(
            backbone=configs.backbone,
            type=configs.type,
            fc_dims=[10],
            intermediate=None,
            backbone_drop_last=True,
        )
        super().__init__(configs=parent_configs, classes=classes, attributes=attributes)
        self.__delattr__("backbone")
        self.__delattr__("head")
        self.configs = configs
        self.classes = classes
        self.loss_fn = GLOLosses(configs=self.configs.losses)

        # Features Extractors
        self.global_feat_ext = split_backbone_levels(
            build_backbone(model_name=self.configs.backbone, drop_last=True).Model,
            prefix="global",
            keywords=[("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4")],
            squeeze=self.configs.squeeze,
        )
        self.local_feat_ext = split_backbone_levels(
            build_backbone(model_name=self.configs.backbone, drop_last=True).Model,
            prefix="local",
            keywords=[("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4")],
            squeeze=self.configs.squeeze,
        )
        self.intrinsic_feat_ext = split_backbone_levels(
            build_informed_backbone(model_name=self.configs.backbone, drop_last=True),
            prefix="intrinsic",
            keywords=[("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4")],
            squeeze=self.configs.squeeze,
        )

        all_is_none = lambda x: all(item is None for item in x)
        all_is_not_none = lambda x: all(item is not None for item in x)
        assertion_items = [
            self.configs.global_decoder[-1].out_channels,
            self.configs.local_attributes_decoder[-1].out_channels,
            self.configs.intrinsic_attributes_decoder[-1].out_channels,
        ]
        assert all_is_not_none(assertion_items) or all_is_none(assertion_items)

        # Decoders
        if self.configs.global_decoder[-1].out_channels:
            self.is_multi_head = False
        else:
            self.is_multi_head = True
            self.configs.global_decoder[-1].out_channels = len(self.classes) + 5
        self.global_decoder = build_conv2d_network(self.configs.global_decoder).Model
        self.mask_decoder = build_conv2d_network(self.configs.mask_decoder).Model
        if not self.configs.category_embedding[-1].out_channels:
            self.configs.category_embedding[-1].out_channels = len(self.classes)
        self.category_embedding = build_fc_network(
            self.configs.category_embedding, pre_squeeze=True
        ).Model
        if self.is_multi_head:
            self.local_attributes_decoder = {}
            self.n_attributes = {}
            for cls in classes:
                total = sum(1 for s in attributes if s.split("::")[0] == cls)
                if total > 0:
                    self.n_attributes[cls] = total

            conf = self.configs.local_attributes_decoder
            for cls, value in self.n_attributes.items():
                conf[-1].out_channels = value
                self.local_attributes_decoder.update(
                    {cls: build_fc_network(conf, pre_squeeze=True).Model}
                )
            self.local_attributes_decoder = nn.ModuleDict(self.local_attributes_decoder)
        else:
            self.local_attributes_decoder = build_fc_network(
                self.configs.local_attributes_decoder, pre_squeeze=True
            ).Model

        if self.is_multi_head:
            self.intrinsic_attributes_decoder = {}
            conf = self.configs.intrinsic_attributes_decoder
            for cls, value in self.n_attributes.items():
                conf[-1].out_channels = value
                self.intrinsic_attributes_decoder.update(
                    {cls: build_fc_network(conf, pre_squeeze=True).Model}
                )
            self.intrinsic_attributes_decoder = nn.ModuleDict(self.intrinsic_attributes_decoder)
        else:
            self.intrinsic_attributes_decoder = build_fc_network(
                self.configs.intrinsic_attributes_decoder, pre_squeeze=True
            ).Model

        if not self.configs.descriptor.category_fc[0].in_channels:
            self.configs.descriptor.category_fc[0].in_channels = len(self.classes)
        self.object_descriptor = build_descriptor(self.configs.descriptor).Model

        # Gates
        self.general_gate = build_conv2d_network(self.configs.gate).Model
        self.local_gate = build_conv2d_network(self.configs.gate).Model
        self.intrinsic_gate = build_conv2d_network(self.configs.gate).Model

        self.classifier = build_fc_network(self.configs.classifier, pre_squeeze=True).Model

        if self.configs.interpreter_type is InterpreterType.Multihead:
            if self.is_multi_head:
                self.configs.interpreter.n_classes = len(classes)
                conf = self.configs.interpreter
                interpreters = {}
                for cls, value in self.n_attributes.items():
                    conf.fc_config[-1].out_channels = value
                    interpreters.update({cls: build_fc_network(conf.fc_config).Model})
                self.interpreter = nn.ModuleDict(interpreters)
            else:
                self.interpreter = build_fc_network(self.configs.interpreter.fc_config).Model
            assert (
                self.configs.interpreter.fc_config[-1].activation is None
            ), "Last layer must not have activation"
        else:
            raise NotImplementedError(
                f"This type {self.configs.interpreter_type} is not implemented"
            )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_avg_pool_7 = nn.AdaptiveAvgPool2d(7)
        self.adaptive_avg_pool_28 = nn.AdaptiveAvgPool2d(28)
        assert self.is_sane()

    def is_sane(self):
        # check that modules are included
        children = list(zip(*self.named_children()))[0]
        check_list = [
            "global_feat_ext",
            "local_feat_ext",
            "intrinsic_feat_ext",
            "global_decoder",
            "mask_decoder",
            "category_embedding",
            "local_attributes_decoder",
            "intrinsic_attributes_decoder",
            "object_descriptor",
            "general_gate",
            "local_gate",
            "intrinsic_gate",
            "classifier",
            "interpreter",
        ]
        for module in check_list:
            assert module in children, module

        # activations sanity
        assert "activation" not in self.global_decoder[-1]._modules.keys(), self.global_decoder[-1]

        # dimensions sanity
        assert (
            self.configs.descriptor.category_fc[0].in_channels
            == self.configs.category_embedding[-1].out_channels
        ), f"{self.configs.descriptor.category_fc[0].in_channels} != {self.configs.category_embedding[-1].out_channels}"
        assert (
            self.configs.descriptor.mask_convs[0].in_channels
            == self.configs.mask_decoder[-1].out_channels
        ), f"{self.configs.descriptor.mask_convs[0].in_channels} != {self.configs.mask_decoder[-1].out_channels}"
        if self.configs.interpreter_type is InterpreterType.Multihead:
            assert (
                self.configs.classifier[-1].out_channels
                == self.configs.interpreter.fc_config[0].in_channels
            ), f"{self.configs.classifier[-1].out_channels} != {self.configs.interpreter.fc_config[0].in_channels}"
            assert (
                self.configs.category_embedding[-1].out_channels
                == self.configs.interpreter.n_classes
            ), f"{self.configs.category_embedding[-1].out_channels} != {self.configs.interpreter.n_classes}"
        return True

    def sub_features(
        self, model_items: ModelInputItems, pool: bool
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # extracting features from different layers of different extractors
        general_features_layers = self.global_feat_ext(model_items.images)
        local_features_layers = self.local_feat_ext(model_items.cropped_images)
        intrinsic_features_layers = self.intrinsic_feat_ext(
            model_items.cropped_images, mask=model_items.cropped_masks
        )

        if pool:
            # Pooling the features
            general_features_layers = torch.cat(
                [
                    self.adaptive_avg_pool(general_features)
                    for general_features in general_features_layers
                ],
                dim=1,
            )
            local_features_layers = torch.cat(
                [
                    self.adaptive_avg_pool(local_features)
                    for local_features in local_features_layers
                ],
                dim=1,
            )
            intrinsic_features_layers = torch.cat(
                [
                    self.adaptive_avg_pool(intrinsic_features)
                    for intrinsic_features in intrinsic_features_layers
                ],
                dim=1,
            )
        return general_features_layers, local_features_layers, intrinsic_features_layers

    def features(self, model_items: ModelInputItems) -> torch.Tensor:
        return torch.cat(self.sub_features(model_items, pool=True), dim=1).squeeze(3).squeeze(2)

    def layers_phase1(self, model_items: ModelInputItems) -> GlidePhase1Output:
        (
            general_features_layers,
            local_features_layers,
            intrinsic_features_layers,
        ) = self.sub_features(model_items, pool=False)

        general_features_layers = torch.cat(
            [
                self.adaptive_avg_pool_28(general_features)
                for general_features in general_features_layers
            ],
            dim=1,
        )
        local_features_layers = torch.cat(
            [self.adaptive_avg_pool_28(local_features) for local_features in local_features_layers],
            dim=1,
        )
        local_features_pooled = self.adaptive_avg_pool(local_features_layers)
        intrinsic_features_layers = torch.cat(
            [
                self.adaptive_avg_pool(intrinsic_features)
                for intrinsic_features in intrinsic_features_layers
            ],
            dim=1,
        )

        # Multi-object detection head
        decoded_objects = self.global_decoder(general_features_layers)
        decoded_mask = self.mask_decoder(local_features_layers)
        decoded_category = self.category_embedding(local_features_pooled)
        decoded_local_attributes = (
            self.local_attributes_decoder[model_items.single_class_name](local_features_pooled)
            if self.is_multi_head
            else self.local_attributes_decoder(local_features_pooled)
        )
        decoded_intrinsic_attributes = (
            self.intrinsic_attributes_decoder[model_items.single_class_name](
                intrinsic_features_layers
            )
            if self.is_multi_head
            else self.intrinsic_attributes_decoder(intrinsic_features_layers)
        )

        return GlidePhase1Output(
            PreLogits=None,
            objects=decoded_objects,
            mask=decoded_mask,
            category=decoded_category,
            attributes_local=decoded_local_attributes,
            attributes_intrinsic=decoded_intrinsic_attributes,
        )

    def layers_phase2(self, model_items: ModelInputItems) -> GlidePhase2Output:
        (
            general_features_layers,
            local_features_layers,
            intrinsic_features_layers,
        ) = self.sub_features(model_items, pool=False)

        # generating description
        category_embedding = self.category_embedding(
            torch.cat(
                [
                    self.adaptive_avg_pool(local_features)
                    for local_features in local_features_layers
                ],
                dim=1,
            )
        )
        description = self.object_descriptor(category=category_embedding, mask=model_items.masks)

        # gating the features based on the description
        general_descriptions = self.general_gate(description).split(1, dim=1)
        general_features_layers = sum(
            self.adaptive_avg_pool(interpolate(gd, size=gf.shape[-2:]) * gf)
            for gd, gf in zip(general_descriptions, general_features_layers)
        )
        local_descriptions = self.local_gate(description).split(1, dim=1)
        local_features_layers = sum(
            self.adaptive_avg_pool(interpolate(ld, size=lf.shape[-2:]) * lf)
            for ld, lf in zip(local_descriptions, local_features_layers)
        )
        intrinsic_descriptions = self.intrinsic_gate(description).split(1, dim=1)
        intrinsic_features_layers = sum(
            self.adaptive_avg_pool(interpolate(intd, size=intf.shape[-2:]) * intf)
            for intd, intf in zip(intrinsic_descriptions, intrinsic_features_layers)
        )

        attributes_embedding = self.classifier(
            torch.cat(
                (general_features_layers, local_features_layers, intrinsic_features_layers), dim=1
            )
        )

        if self.configs.interpreter_type is InterpreterType.Multihead:
            prelogits = self.interpreter[model_items.single_class_name](attributes_embedding)
        elif self.configs.interpreter_type is InterpreterType.Projection:
            raise NotImplementedError
        else:
            raise NotImplementedError

        return GlidePhase2Output(
            PreLogits=prelogits,
            category_embedding=category_embedding,
            category_id=model_items.class_ids,
            attributes_embedding=attributes_embedding,
        )

    def inference(self, model_items: ModelInputItems) -> torch.Tensor:
        out = self.layers_phase2(model_items).PreLogits
        assert len(out.shape) == 2
        return softmax(out, dim=1)

    def on_mode_epoch_end(
        self, mode: str, cache: t.Dict[str, t.Any], classes: t.Set
    ) -> t.Dict[str, float]:
        if "phase2/output" in cache.keys():

            sums = {k: sum(v) for k, v in cache.items()}
            # gs = torch.cat(cache["phase2/gt"], dim=0)
            # outs = torch.cat(cache["phase2/output"], dim=0)
            sum_vals = {k.replace("phase2/", ""): v for k, v in sums.items()}
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
            records = {
                f"{mode}/phase2_{k}": v.item() if hasattr(v, "item") else v
                for k, v in records.items()
            }
            return records
        else:
            return {}


class GLOLosses:
    def __init__(self, configs: GlideLossesConfig):
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        # self.bce_with_logits_loss_reduction_sum = nn.BCEWithLogitsLoss(reduction="sum")
        self.multi_class_loss = nn.CrossEntropyLoss()
        self.configs = configs

    def global_decoder_losses(
        self,
        output: GlidePhase1Output,
        model_items: ModelInputItems,
        configs: t.Optional[GlideLossesConfig] = None,
    ) -> GeneralLoss:
        configs = configs if configs else self.configs
        instance_exist = model_items.instances_tensor[:, 0:1]
        global_decoder_pred = output.objects
        losses = GeneralLoss()
        if (instance_exist == 1).any():
            indices = instance_exist == 1
            losses.add(
                LossItem(
                    name="loss/phase1/global_decoder/confidence",
                    value=self.bce_with_logits_loss(global_decoder_pred[:, 0:1], instance_exist),
                    weight=configs.weights["global_decoder_confidence"],
                    isBackpropagated=True,
                    isLogged=True,
                    isCached=False,
                )
            )
            losses.add(
                LossItem(
                    name="loss/phase1/global_decoder/position",
                    value=self.mse_loss(
                        global_decoder_pred[:, 1:3][indices.repeat(1, 2, 1, 1)].sigmoid(),
                        model_items.instances_tensor[:, 1:3][indices.repeat(1, 2, 1, 1)],
                    ),
                    weight=configs.weights["global_decoder_position"],
                    isBackpropagated=True,
                    isLogged=True,
                    isCached=False,
                )
            )
            losses.add(
                LossItem(
                    name="loss/phase1/global_decoder/dimensions",
                    value=self.mse_loss(
                        global_decoder_pred[:, 3:5][indices.repeat(1, 2, 1, 1)].exp(),
                        model_items.instances_tensor[:, 3:5][indices.repeat(1, 2, 1, 1)],
                    ),
                    weight=configs.weights["global_decoder_dimensions"],
                    isBackpropagated=True,
                    isLogged=True,
                    isCached=False,
                )
            )
            losses.add(
                LossItem(
                    name="loss/phase1/global_decoder/category",
                    value=self.multi_class_loss(
                        global_decoder_pred[:, 5:]
                        .transpose(0, 1)[:, indices.squeeze(1)]
                        .transpose(0, 1),
                        model_items.instances_tensor[:, 5:].to(dtype=torch.int64)[indices],
                    ),
                    weight=configs.weights["global_decoder_category"],
                    isBackpropagated=True,
                    isLogged=True,
                    isCached=False,
                )
            )
        return losses

    def attributes_loss(
        self,
        output_attributes: torch.Tensor,
        P: torch.Tensor,
        N: torch.Tensor,
        name: str,
        weight: float,
    ):
        losses = GeneralLoss()
        losses.add(
            LossItem(
                name=f"{name}/attributes/positive",
                value=self.bce_with_logits_loss(
                    output_attributes[P], torch.ones([P.sum()]).to(device=P.device)
                )
                if P.any()
                else torch.tensor(0.0, requires_grad=True).to(device=P.device),
                weight=weight,
                isBackpropagated=True,
                isCached=False,
                isLogged=True,
            )
        )
        losses.add(
            LossItem(
                name=f"{name}/attributes/negative",
                value=self.bce_with_logits_loss(
                    -output_attributes[N], torch.ones([N.sum()]).to(device=N.device)
                )
                if N.any()
                else torch.tensor(0.0, requires_grad=True).to(device=N.device),
                weight=weight,
                isBackpropagated=True,
                isCached=False,
                isLogged=True,
            )
        )
        return losses

    def __call__(
        self,
        output: t.Union[GlidePhase1Output, GlidePhase2Output],
        model_items: ModelInputItems,
        configs: t.Optional[GlideLossesConfig] = None,
    ) -> GeneralLoss:
        configs = configs if configs else self.configs
        output_loss = GeneralLoss()

        if isinstance(output, GlidePhase1Output):
            """
            It should contain losses for the following:
                1. Global Decoder: generate all objects in the image back.
                2. Mask Decoder: generate the mask.
                3. Category Embedding: 1 hot encoder of the category type.
                4. Local Attributes and Intrinsic Attributes: 1 hot encoder each.
            """
            output_loss.extend(
                self.global_decoder_losses(output=output, model_items=model_items, configs=configs)
            )
            output_loss.add(
                LossItem(
                    name="loss/phase1/local_decoder/mask",
                    value=self.bce_with_logits_loss(output.mask, model_items.cropped_masks),
                    weight=configs.weights["local_decoder_mask"],
                    isBackpropagated=True,
                    isCached=False,
                    isLogged=True,
                )
            )
            output_loss.add(
                LossItem(
                    name="loss/phase1/local_decoder/category",
                    value=self.multi_class_loss(output.category, model_items.class_ids),
                    weight=configs.weights["local_decoder_category"],
                    isBackpropagated=True,
                    isCached=False,
                    isLogged=True,
                )
            )

            gt_attributes = model_items.attributes_labels.to(dtype=torch.float32)
            P = gt_attributes == 1  # Positive
            N = gt_attributes == -1  # Negative
            U = gt_attributes == 0  # Unlabeled
            assert P.sum() + N.sum() + U.sum() == gt_attributes.numel()

            output_loss.extend(
                self.attributes_loss(
                    output.attributes_local,
                    P,
                    N,
                    "loss/phase1/local_decoder",
                    configs.weights["local_decoder_attributes"],
                )
            )
            output_loss.extend(
                self.attributes_loss(
                    output.attributes_intrinsic,
                    P,
                    N,
                    "loss/phase1/intrinsic_decoder",
                    configs.weights["intrinsic_decoder_attributes"],
                )
            )

            # calculating some metrics
            def accuracy(pred):
                TNs = pred[N] <= 0.0  # True Negative
                TPs = pred[P] > 0.0
                FNs = pred[P] <= 0.0
                FPs = pred[N] > 0.0  # False Positive
                assert TPs.sum() + FNs.sum() == P.sum(), "some positive labels are skipped"
                assert TNs.sum() + FPs.sum() == N.sum(), "some negative labels are skipped"
                return (
                    (TPs.sum() + TNs.sum()) / (P.sum() + N.sum()) if (P.sum() + N.sum()) > 0 else 1
                )

            def iou(gts, preds):
                gts = gts >= 0.5
                preds = preds.sigmoid() >= 0.5
                ious = []
                for gt, pred in zip(gts, preds):
                    intersection = torch.bitwise_and(gt, pred).sum()
                    union = torch.bitwise_or(gt, pred).sum()
                    assert (
                        intersection <= union
                    ), f"intersection is {intersection} and union is {union}"
                    ious.append(intersection / union)
                return sum(ious) / len(ious)

            metrics_dict = {
                "local/attributes/accuracy": accuracy(output.attributes_local),
                "intrinsic/attributes/accuracy": accuracy(output.attributes_intrinsic),
                "mask/iou": iou(model_items.cropped_masks, output.mask),
                "category/accuracy": sum(output.category.argmax(dim=1) == model_items.class_ids)
                / len(output.category),
            }
            for k, v in metrics_dict.items():
                output_loss.add(
                    LossItem(
                        name=f"phase1/{k}",
                        value=v,
                        isLogged=True,
                        isCached=True,
                        isBackpropagated=False,
                    )
                )

            output_loss.extend(
                [
                    LossItem(
                        name="phase1/gt",
                        value=gt_attributes,
                        isLogged=False,
                        isCached=False,
                        isBackpropagated=False,
                    ),
                    LossItem(
                        name="phase1/output/local",
                        value=output.attributes_local,
                        isLogged=False,
                        isCached=False,
                        isBackpropagated=False,
                    ),
                    LossItem(
                        name="phase1/output/intrinsic",
                        value=output.attributes_intrinsic,
                        isLogged=False,
                        isCached=False,
                        isBackpropagated=False,
                    ),
                ]
            )

            # drawing binary masks
            w = output.mask.shape[-1]
            mask_grid = ToPILImage()(
                torch.cat(
                    (
                        output.mask.sigmoid(),
                        (output.mask.sigmoid() >= 0.5).to(dtype=torch.int),
                        model_items.cropped_masks,
                    ),
                    dim=3,
                ).view(-1, 3 * w)
            ).convert("L")
            output_loss.add(
                LossItem(
                    name="phase1_mask[prediction|thresholded|GT]",
                    value=mask_grid,
                    isBackpropagated=False,
                    isCached=False,
                    isLogged=False,
                    isImage=True,
                )
            )

            # drawing polygons
            def generate_boxes(objects, activate: bool):
                H, W = objects.shape[-2:]
                if activate:
                    exists = objects[0, 0].sigmoid() >= 0.5
                else:
                    exists = objects[0, 0] > 0.5
                cxs, cys, widths, heights = objects[0, 1:5]
                scale = 8
                mask = Image.new("1", (W * scale, H * scale))
                draw = ImageDraw.Draw(mask)
                for ih in range(H):
                    for iw in range(W):
                        if exists[ih, iw]:
                            cx, cy, width, height = (
                                cxs[ih, iw],
                                cys[ih, iw],
                                widths[ih, iw],
                                heights[ih, iw],
                            )
                            if activate:
                                cx, cy = cx.sigmoid(), cy.sigmoid()
                                width, height = width.exp(), height.exp()
                            pts = [
                                ((iw + cx - width / 2) * scale, (ih + cy - height / 2) * scale),
                                ((iw + cx - width / 2) * scale, (ih + cy + height / 2) * scale),
                                ((iw + cx + width / 2) * scale, (ih + cy + height / 2) * scale),
                                ((iw + cx + width / 2) * scale, (ih + cy - height / 2) * scale),
                            ]
                            draw.polygon(pts, fill=False, outline=True)
                            if activate:
                                draw.text(
                                    ((iw + cx) * scale, (ih + cy) * scale),
                                    f"{objects[0, 5:, ih, iw].argmax()}",
                                    font=ImageFont.load_default(),
                                )
                            else:
                                draw.text(
                                    ((iw + cx) * scale, (ih + cy) * scale),
                                    f"{objects[0, 5, ih, iw]}",
                                    font=ImageFont.load_default(),
                                )
                return ToTensor()(mask)

            output_boxes = generate_boxes(output.objects, True)
            gt_boxes = generate_boxes(model_items.instances_tensor, False)
            mask_grid = ToPILImage()(torch.cat((output_boxes, gt_boxes), dim=2)).convert("L")
            output_loss.add(
                LossItem(
                    name="phase1/global_detected_boxes",
                    value=mask_grid,
                    isBackpropagated=False,
                    isCached=False,
                    isLogged=False,
                    isImage=True,
                )
            )

        elif isinstance(output, GlidePhase2Output):
            """
            It should contain losses for the following:
                1. Attributes prediction after passing through the interpreter.
                2. Category Embedding: 1 hot encoder of the category type.
            """
            gt_attributes = model_items.attributes_labels.to(dtype=torch.float32)
            P = gt_attributes == 1  # Positive
            N = gt_attributes == -1  # Negative
            U = gt_attributes == 0  # Unlabeled
            assert P.sum() + N.sum() + U.sum() == gt_attributes.numel()

            output_loss.extend(
                self.attributes_loss(
                    output.PreLogits, P, N, "loss/phase2", configs.weights["phase2_attributes"]
                )
            )
            output_loss.add(
                LossItem(
                    name="loss/phase2/category",
                    value=self.multi_class_loss(output.category_embedding, model_items.class_ids),
                    weight=configs.weights["phase2_category"],
                    isBackpropagated=True,
                    isCached=False,
                    isLogged=True,
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
                "accuracy": (TP.sum() + TN.sum()) / (P.sum() + N.sum())
                if (P.sum() + N.sum()) > 0
                else 1,
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
                        name=f"phase2/{k}",
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
                        name=f"phase2/{k}",
                        value=v,
                        isLogged=False,
                        isCached=True,
                        isBackpropagated=False,
                    )
                )

        else:
            raise NotImplementedError(f"Unknown phase type {output}")

        output_loss.add(
            LossItem(
                name="phase_number",
                value=1 if isinstance(output, GlidePhase1Output) else 2,
                isLogged=True,
                isCached=False,
                isBackpropagated=False,
                isImage=False,
            )
        )
        return output_loss


if __name__ == "__main__":
    import yaml

    from dataset.car.dataset import CARDataset
    from structures.common import Hyperparameters
    from structures.model import ModelInputItems

    with open("/home/krm/models/car_attributes_2d/configs/models/vaw/glidenet.yaml") as f:
        conf = Hyperparameters.from_dict(yaml.safe_load(f))
    dataset = CARDataset(conf.dataset, "train")
    model = Glidenet(
        configs=conf.model,
        classes=dataset.codec.categories_map,
        attributes=dataset.codec.attributes_set,
    )
    model.layers_phase1(ModelInputItems.collate([dataset[0]]))
    model.layers_phase2(ModelInputItems.collate([dataset[0]]))
    model.features(ModelInputItems.collate([dataset[0]]))
    for sample in dataset:
        batch = ModelInputItems.collate([sample])
        out = model.layers_phase1(batch)
        print(out)
