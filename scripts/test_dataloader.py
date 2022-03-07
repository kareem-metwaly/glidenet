import yaml

from structures.common import Hyperparameters
from trainer.train import SubtaskDataModule


def no_op(x):
    return x


def main():
    with open("configs/models/car/glidenet.yaml", "r") as f:
        hparams = yaml.safe_load(f)
        hp = Hyperparameters.from_dict(hparams)

    print("Loading data module")
    data_module = SubtaskDataModule(hp.dataset, rank=0, world_size=2)
    dl = data_module.get_train_dataloader()

    for batch in dl:
        project_ids = set(b["project_id"] for b in batch)
        print(project_ids)
        assert len(project_ids) == 1


if __name__ == "__main__":
    main()
