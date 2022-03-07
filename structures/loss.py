import typing as t
from dataclasses import dataclass

import torch
from PIL.Image import Image
from torch import Tensor


@dataclass
class LossItem:
    name: str = ""
    value: t.Any = None
    isBackpropagated: bool = False
    isCached: bool = False
    isLogged: bool = False
    isImage: bool = False
    weight: t.Optional[t.Union[float, Tensor]] = 1.0

    def __post_init__(self):
        if self.isBackpropagated:
            assert torch.is_floating_point(self.value), f"{self.name} has type {type(self.value)}"
            assert self.value.numel() == 1, self.value.shape
            assert self.value.requires_grad or not torch.is_grad_enabled()
            assert not self.isImage
            assert not (
                self.value.isnan() or self.value.isinf()
            ), f"loss value is nan or inf; {self.value} for {self.name}"
        if self.isImage:
            assert isinstance(self.value, Image)
            self.name = self.name.replace("/", "_").replace(
                "\\", "_"
            )  # as it may be understood as folder


class GeneralLoss:
    _items: t.List[LossItem]

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            self._items = args[0]
        elif len(kwargs) == 1 and len(args) == 0:
            self._items = kwargs.popitem()[1]
        elif len(kwargs) == 0 and len(args) == 0:
            self._items = []
        else:
            raise SyntaxError

    @property
    def names(self) -> t.Iterator[str]:
        return (item.name for item in self._items)

    @property
    def BackpropagatedItems(self) -> t.Iterator[LossItem]:
        return (item for item in self._items if item.isBackpropagated)

    @property
    def CachedItems(self) -> t.Iterator[LossItem]:
        return (item for item in self._items if item.isCached)

    @property
    def LoggedItems(self) -> t.Iterator[LossItem]:
        return (item for item in self._items if item.isLogged)

    @property
    def LoggedImages(self) -> t.Iterator[LossItem]:
        return (item for item in self._items if item.isImage)

    def __repr__(self):
        string = [f"{item.name}={item.value}" for item in self._items if item.isBackpropagated]
        return f"{self.__class__.__name__}({string})"

    @property
    def value(self) -> Tensor:
        """
        Should return the value that will be used later for backward propagation
        """
        total = 0
        for item in self.BackpropagatedItems:
            total = total + item.weight * item.value
        return total

    @property
    def logs(self) -> t.Optional[t.Dict[str, float]]:
        """
        Should return a dict of values to be logged in mlflow
        """
        items = {item.name: item.value for item in self.LoggedItems}
        items.update({k: v.item() for k, v in items.items() if hasattr(v, "item")})
        return items

    @property
    def cache(self) -> t.Optional[t.Dict[str, t.Union[float, Tensor]]]:
        """
        should return values to be cached and then averaged over each epoch
        """
        items = {item.name: item.value for item in self.CachedItems}
        for k, v in items.items():
            if hasattr(v, "item") and v.numel() == 1:
                items.update({k: v.item()})
            elif hasattr(v, "detach"):
                items.update({k: v.detach()})
        return items

    def add(self, loss_item: LossItem) -> None:
        assert loss_item.name not in self.names, f"{loss_item.name} and {list(self.names)}"
        self._items.append(loss_item)

    def remove(self, name: str) -> None:
        idx = [item.name for item in self._items].index(name)
        self._items.pop(idx)

    def extend(self, losses: t.Union["GeneralLoss", t.Sequence[LossItem]]) -> None:
        for loss_item in losses:
            self.add(loss_item)

    def __iter__(self):
        for item in self._items:
            yield item

    def update_existing(
        self,
        new_loss_item: LossItem = None,
        new_loss_items: t.Union[t.List[LossItem], "GeneralLoss"] = None,
    ) -> None:
        """
        update an already existing item or list of items
        """
        assert (new_loss_item is not None or new_loss_items is not None) and (
            new_loss_items is not None and new_loss_item is not None
        ), f"only one value can be set, new_loss_item or new_loss_items; given {new_loss_item} and {new_loss_items}"
        if new_loss_item is not None:
            new_loss_items = [new_loss_item]
        for item in new_loss_items:
            self.remove(item.name)
            self.add(item)


class VAWPaperLoss(GeneralLoss):
    pass


@dataclass
class VAWLossesConfig:
    lambda_div: float
    lambda_rel: float
    w: float
