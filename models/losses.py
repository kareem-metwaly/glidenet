import torch
from torch.nn import functional as F

from structures.abstract_classes import ABCModelOutput
from structures.loss import GeneralLoss, LossItem
from structures.model import ModelInputItems


def calc_losses(
    output: ABCModelOutput,
    model_items: ModelInputItems,
) -> GeneralLoss:
    """
    Compute the losses for VAW models
    """
    class_ids = model_items.class_ids  # assuming all belonging to the same class
    if len(set(idx.item() for idx in class_ids)) > 1:
        raise NotImplementedError(
            f"currently, it supports one class_id for loss calculations, {class_ids}"
        )

    class_id = class_ids[0]

    output_loss = GeneralLoss()
    output_loss.add(
        LossItem(
            name="class_id",
            value=class_id,
            isLogged=True,
            isCached=False,
            isBackpropagated=False,
            weight=1.0,
        )
    )

    gt_attributes = model_items.attributes_labels.to(dtype=torch.float32)
    pred_prelogits = output.PreLogits

    P = gt_attributes == 1  # Positive
    N = gt_attributes == -1  # Negative
    U = gt_attributes == 0  # Unlabeled
    assert P.sum() + N.sum() + U.sum() == gt_attributes.numel()

    pos_output = torch.where(P, pred_prelogits, torch.ones_like(pred_prelogits).fill_(-1e10))
    neg_output = torch.where(N, -pred_prelogits, torch.ones_like(pred_prelogits).fill_(-1e10))

    # Calculating losses for backward propagation
    losses = {
        "BCE_positive": F.binary_cross_entropy_with_logits(
            pos_output, P.to(dtype=torch.float32), reduction="mean"
        ),
        "BCE_negative": F.binary_cross_entropy_with_logits(
            neg_output, N.to(dtype=torch.float32), reduction="mean"
        ),
    }
    [
        output_loss.add(
            LossItem(
                name=k, value=v, isLogged=True, isCached=False, isBackpropagated=True, weight=1.0
            )
        )
        for k, v in losses.items()
    ]

    # calculating some metrics
    Labeled = P.bitwise_or(N)
    TN = pred_prelogits[N] <= 0.0  # True Negative
    TP = pred_prelogits[P] > 0.0
    FN = pred_prelogits[P] <= 0.0
    FP = pred_prelogits[N] > 0.0  # False Positive
    assert TP.sum() + FN.sum() == P.sum(), "some positive labels are skipped"
    assert TN.sum() + FP.sum() == N.sum(), "some negative labels are skipped"
    PP = pred_prelogits[Labeled] > 0.0  # Predicted Positive (from labeled data P+N only)
    PN = pred_prelogits[Labeled] <= 0.0  # Predicted Negative (from labeled data P+N only)
    PPU = pred_prelogits[U] > 0.0  # unlabeled predicted positive
    PNU = pred_prelogits[U] <= 0.0  # unlabeled predicted negative

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
    [
        output_loss.add(
            LossItem(
                name=k, value=v, isLogged=True, isCached=False, isBackpropagated=False, weight=1.0
            )
        )
        for k, v in metrics_dict.items()
    ]

    # values to be used for average metrics
    # gts, outs = zip(*[(gt_attributes[i, idx].detach().cpu().numpy(), output[i, idx].detach().cpu().numpy()) for i, idx in enumerate(Labeled)])
    extra_dict = {
        f"TP_{class_id}": TP.sum(),
        f"TN_{class_id}": TN.sum(),
        f"P_{class_id}": P.sum(),
        f"N_{class_id}": N.sum(),
        f"PP_{class_id}": PP.sum(),
        f"PN_{class_id}": PN.sum(),
        "output": pred_prelogits,
        "gt": gt_attributes,
        # "AveragePrecision": np.mean([metrics.average_precision_score(gt, out) for gt, out in zip(gts, outs)]),
    }
    [
        output_loss.add(
            LossItem(
                name=k, value=v, isLogged=False, isCached=True, isBackpropagated=False, weight=1.0
            )
        )
        for k, v in extra_dict.items()
    ]

    return output_loss
