import json
import os
import tempfile
import typing as t

import mlflow
import pandas as pd
import sh
import smart_open
import torch
import yaml

from models.subtask_model import SubtaskModel
from structures.common import Hyperparameters


ClassesAttributes = t.NamedTuple(
    "ClassesAttributes", (("Classes", t.List[str]), ("Attributes", t.List[str]))
)


def calculate_mean_average_precision(outputs: torch.Tensor, gts: torch.Tensor):
    """
    Loop for each attribute and calculate the average precision (AP)
    AP is calculated by sum over positive samples / number of positive samples.
    We sum the precision for best k images
    Args:
        outputs (torch.Tensor):
        gts (torch.Tensor):

    Returns:

    """
    assert (
        outputs.shape == gts.shape
    ), f"Mismatch between sizes of outputs ({outputs.shape}) and gts ({gts.shape})"
    assert len(outputs.shape) == 2, outputs.shape
    max_vals, max_ids = outputs.topk(len(outputs), dim=0)
    n_samples, n_attributes = outputs.shape
    APs = []
    for attr_idx in range(n_attributes):
        attr_precisions = []
        for sample_idx in range(n_samples):
            if max_vals[sample_idx, attr_idx] <= 0.0:
                break
            if gts[max_ids[sample_idx, attr_idx], attr_idx] == 1:
                topk_output = outputs[max_ids[: sample_idx + 1, attr_idx], attr_idx]
                topk_gt = gts[max_ids[: sample_idx + 1, attr_idx], attr_idx]
                gt_P = topk_gt == 1
                TP = (topk_output[gt_P] > 0.0).sum()  # True Positive
                PP = len(topk_output[gt_P])  # Predicted Positive
                attr_precisions.append((TP / PP).item())
            elif gts[max_ids[sample_idx, attr_idx], attr_idx] == -1:
                # we only consider labeled samples
                attr_precisions.append(0.0)
        if len(attr_precisions) > 0:
            APs.append(sum(attr_precisions) / len(attr_precisions))
    mAP = sum(APs) / len(APs)
    return mAP


def load_state_dict(
    model: SubtaskModel,
    state_dict: t.Mapping[str, t.Any],
    model_weights_key: str = "model",
    remove_weights_key_prefix: str = "model.",
    add_weights_key_prefix: str = "",
    strict: bool = True,
):
    """Load state_dict to the model

    Parameters
    ----------
    model : SubtaskModel
        model
    state_dict : t.Mapping[str, t.Any]
        state_dict
    model_weights_key : str
        model_weights_key
    remove_weights_key_prefix : str
        Set to "module." if the model is trained with DDP. Otherwise set to
        "model."
    add_weights_key_prefix: str
        Appended to the beginning of the state dict.
    strict : bool
        strict
    """
    model_state_dict = {
        f"{add_weights_key_prefix}{k.replace(remove_weights_key_prefix, '')}": v
        for k, v in state_dict[model_weights_key].items()
    }
    k1, k2 = model.model.load_state_dict(model_state_dict, strict=strict)
    if k1:
        logger.info("Missing keys")
        logger.info(sorted(k1))
    if k2:
        logger.info("Unexpected keys")
        logger.info(sorted(k2))


def load_classes_attributes(
    model_name: t.Optional[str] = None, path_override: t.Optional[t.Tuple[str, str]] = None
) -> ClassesAttributes:
    if path_override is None:
        path_override = [
            os.path.join(BASE_MODEL_DIR, model_name, f"{idx}.json")
            for idx in ["classes", "attributes"]
        ]
    with smart_open.open(path_override[0]) as f:
        classes = json.load(f)
    with smart_open.open(path_override[1]) as f:
        attributes = json.load(f)
    return ClassesAttributes(Classes=classes, Attributes=attributes)


def load_hparams(
    model_name: t.Optional[str] = None, path_override: t.Optional[str] = None
) -> Hyperparameters:
    assert (model_name is None) ^ (
        path_override is None
    ), "Exactly one of model_name, path_override must be specified"
    if path_override is None:
        model_dir = os.path.join(BASE_MODEL_DIR, model_name)  # type: ignore
        path_override = os.path.join(model_dir, "hparams.yaml")
    with smart_open.open(path_override, "r") as f:
        hparams = yaml.safe_load(f)
    hp = Hyperparameters.from_dict(hparams)
    return hp


def load_df(model_name: str) -> t.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with smart_open.open(os.path.join(BASE_MODEL_DIR, model_name, "df_train.csv"), "r") as f:
        df_train = pd.read_csv(f)
    with smart_open.open(os.path.join(BASE_MODEL_DIR, model_name, "df_val.csv"), "r") as f:
        df_val = pd.read_csv(f)
    with smart_open.open(os.path.join(BASE_MODEL_DIR, model_name, "df_test.csv"), "r") as f:
        df_test = pd.read_csv(f)
    return df_train, df_val, df_test


def load_trained_model(
    model_name: str,
    TaskModel: t.Type[SubtaskModel],
    map_location: str = "cpu",
    model_weights_key: str = "model",
    remove_weights_key_prefix: str = "model.",
    task_model_kwargs: t.Optional[t.Mapping] = None,
) -> t.Tuple[Hyperparameters, SubtaskModel]:
    """Load trained model

    Parameters
    ----------
    model_name : str
        model_name
    TaskModel : t.Type[SubtaskModel]
        TaskModel
    map_location : str
        map_location
    model_weights_key : str
        model_weights_key
    remove_weights_key_prefix : str
        Set to "module." if the model is trained with DDP. Otherwise set to
        "model."
    task_model_kwargs : t.Optional[t.Mapping]
        task_model_kwargs

    Returns
    -------
    t.Tuple[Hyperparameters, SubtaskModel]

    """
    if task_model_kwargs is None:
        task_model_kwargs = {}

    model_dir = os.path.join(BASE_MODEL_DIR, model_name)

    hp = load_hparams(model_name)

    state_dict_fpath = os.path.join(model_dir, "checkpoints", "last.ckpt")
    # smart_open.open(state_dict_fpath) directly is really slow so copy the
    # state_dict to a temporary location
    with tempfile.NamedTemporaryFile() as f:
        # pylint: disable=no-member
        sh.aws.s3.cp(state_dict_fpath, f.name)
        with open(f.name, "rb") as f:
            state_dict = torch.load(f, map_location=map_location)

    classes_attributes = load_classes_attributes(model_name)
    model = TaskModel(
        hp,
        classes=classes_attributes.Classes,
        attributes=classes_attributes.Attributes,
        **task_model_kwargs,
    )
    load_state_dict(
        model,
        state_dict,
        model_weights_key=model_weights_key,
        remove_weights_key_prefix=remove_weights_key_prefix,
    )
    model.eval()
    return hp, model


def load_model_state_dict_from_mlflow(
    TaskModel: t.Type[SubtaskModel],
    model_name: str,
    model_version: int,
    map_location: str = "cpu",
    remove_weights_key_prefix: str = "model.",
    task_model_kwargs: t.Optional[t.Mapping] = None,
) -> t.Tuple[Hyperparameters, SubtaskModel]:
    """
    Load a model from mlflow's model registry.
    Args:
        TaskModel (t.Type[SubtaskModel]): class of the SubtaskModel
        model_name: Name in mlflow's model registry
        model_version: Version in mlflow's model registry
        map_location:
        remove_weights_key_prefix:
        task_model_kwargs:
    Returns:
        (hparams, model: SubtaskModel)
    """
    if task_model_kwargs is None:
        task_model_kwargs = {}

    client = mlflow.tracking.MlflowClient()

    mlflow_run_id = find_run_for_model_version(model_name, model_version)

    local_hparams_f = client.download_artifacts(mlflow_run_id, "hparams.yaml")
    local_classes_attributes_f = (
        client.download_artifacts(mlflow_run_id, "classes.json"),
        client.download_artifacts(mlflow_run_id, "attributes.json"),
    )

    hp = load_hparams(path_override=local_hparams_f)
    classes_attributes = load_classes_attributes(path_override=local_classes_attributes_f)

    state_dict = load_pytorch_state_dict_from_model_version(
        model_name, model_version, map_location=map_location
    )
    model = TaskModel(
        hp,
        classes=classes_attributes.Classes,
        attributes=classes_attributes.Attributes,
        **task_model_kwargs,
    )
    wrapped_state_dict = dict(
        model=state_dict
    )  # dumb hack to get state_dict into format usable by load_state_dict
    load_state_dict(
        model,
        wrapped_state_dict,
        model_weights_key="model",
        remove_weights_key_prefix=remove_weights_key_prefix,
    )
    model.eval()
    return model.hparams, model


def load_mlflow_model_with_fallback(
    TaskModel,
    model_name: str,
    model_weights_key: str,
    remove_weights_key_prefix: str,
    mlflow_model_name: t.Optional[str] = None,
    mlflow_model_version: t.Optional[int] = None,
) -> t.Tuple[Hyperparameters, SubtaskModel]:
    """
    If mlflow_model_name/mlflow_model_version are provided (and aren't None),
    this will try to load the resulting model from the MLFlow model registry,
    and fall back to loading model weights from s3 if mlflow loading fails.
    Otherwise, this will load model weights from s3.

    Args:
        TaskModel: Class of model
        model_name: Name of model in s3
        model_weights_key:
        remove_weights_key_prefix:
        mlflow_model_name: Name of model in mlflow's model registry
        mlflow_model_version: Version of model in mlflow's model registry

    Returns:
        (hparams, model)
    """
    # TODO remove fallback once we've migrated completely to using mlflow to store models
    try:
        if mlflow_model_name is None or mlflow_model_version is None:
            # Loading from mlflow would fail anyway, might as well jump to fallback now
            raise ValueError(
                "At least one of mlflow_model_name or mlflow_model_version is not provided"
            )
        logger.info(f"Loading mlflow model {mlflow_model_name} ver.{mlflow_model_version}")
        mlflow_hp, mlflow_model = load_model_state_dict_from_mlflow(
            TaskModel,
            mlflow_model_name,
            mlflow_model_version,
        )
        hp, model = mlflow_hp, mlflow_model
        logger.info("MLFlow model loaded")
    except:  # noqa: E722
        logger.warning("MLFlow model unsuccessfully loaded; falling back to old models")
        logger.info(f"Loading model {model_name}")
        hp, model = load_trained_model(
            model_name,
            TaskModel,
            map_location="cpu",
            model_weights_key=model_weights_key,
            remove_weights_key_prefix=remove_weights_key_prefix,
        )
        logger.info("Model loaded")
    model = model.eval()
    return hp, model
