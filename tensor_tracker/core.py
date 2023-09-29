# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Utility for tracking activations and gradients at `nn.Module` outputs.

Use `track` to start tracking a module & submodule. Then use the original module
as usual. Your `Tracker` will be filled with a list of `Stash`es, containing
copies of fwd/bwd tensors at (sub)module outputs. (Beware, this can consume
a lot of memory.)

Usage:

```
with tensor_tracker.track(model) as tracker:
    model(inputs).backward()

print(list(tracker))
# => [Stash(name="0.linear", type=nn.Linear, grad=False, value=tensor(...)),
#     ...]

display(tracker.to_frame())  # requires 'pandas'
```

Advanced usage:

 - Filter modules based on name:
   `track(include="<regex>", exclude="<regex>")`

 - Pre-transform tracked tensors to save memory:
   `track(stash_value=lambda t: t.std().detach().cpu())`

 - Customise tracked state:
   `track(stash=lambda event: ...)`

 - Manually register/unregister hooks:
  `tracker = Tracker(); tracker.register(...); tracker.unregister()`

See also: example of [visualising transformer activations & gradients using UMAP](example.html).
"""

import dataclasses
import re
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
)

import torch.utils.hooks
from torch import Tensor, nn


@dataclass
class Event:
    name: str
    type: Type[nn.Module]
    grad: bool
    value: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class Stash:
    name: str
    type: Type[nn.Module]
    grad: bool
    value: Any  # output(s) or grad_output(s)

    @property
    def first_value(self) -> Any:
        def _value(v: Any) -> Any:
            if isinstance(v, (tuple, list)) and len(v) >= 1:
                return _value(v[0])
            return v

        return _value(self.value)


StashFn = Callable[[Event], Stash]
StashValueFn = Callable[[Tensor], Any]


def rmap_tensor(value: Any, fn: Callable[[Tensor], Any]) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(rmap_tensor(a, fn) for a in value)
    if isinstance(value, dict):
        return {rmap_tensor(k, fn): rmap_tensor(a, fn) for k, a in value.items()}
    if dataclasses.is_dataclass(value):
        return type(value)(**{k: rmap_tensor(v, fn) for k, v in value.__dict__.items()})
    if isinstance(value, Tensor):
        return fn(value)
    return value


def default_stash_value(tensor: Tensor) -> Tensor:
    return tensor.detach().cpu().clone()


def default_stash(event: Event, stash_value: StashValueFn) -> Stash:
    return Stash(
        event.name, event.type, event.grad, rmap_tensor(event.value, stash_value)
    )


def get_stash_fn(
    stash_value: Optional[StashValueFn] = None, stash: Optional[StashFn] = None
) -> StashFn:
    if stash_value and stash:
        raise ValueError("Cannot provide StashValueFn and StashFn to get_stash_fn()")
    if stash:
        return stash
    return partial(default_stash, stash_value=stash_value or default_stash_value)


NamePattern = Union[None, Pattern[str], str]


class Tracker:
    def __init__(self, stash: StashFn):
        self.stashes: List[Stash] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._stash = stash

    # Registration/tracking

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.unregister()

    def clear(self) -> None:
        self.stashes.clear()

    def register(self, module: nn.Module, name: str = "", grad: bool = True) -> None:
        self._handles.append(
            module.register_forward_hook(
                partial(self._forward_hook, name=name), with_kwargs=True
            )
        )
        if grad:
            self._handles.append(
                module.register_full_backward_pre_hook(
                    partial(self._backward_hook, name=name)
                )
            )

    def register_all(
        self,
        module: nn.Module,
        grad: bool = True,
        include: NamePattern = None,
        exclude: NamePattern = None,
    ) -> None:
        include = re.compile(include) if isinstance(include, str) else include
        exclude = re.compile(exclude) if isinstance(exclude, str) else exclude
        for name, child in module.named_modules():
            if ((not include) or include.search(name)) and not (
                exclude and exclude.search(name)
            ):
                self.register(child, name, grad=grad)

    def unregister(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _forward_hook(
        self,
        module: nn.Module,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        output: Any,
        *,
        name: str,
    ) -> None:
        self.stashes.append(
            self._stash(Event(name, type(module), False, output, args, kwargs))
        )

    def _backward_hook(self, module: nn.Module, grad_output: Any, *, name: str) -> None:
        self.stashes.append(
            self._stash(Event(name, type(module), True, grad_output, (), {}))
        )

    # Read results

    def __str__(self) -> str:
        return f"Tracker(stashes={len(self)}, tracking={len(self._handles)})"

    def __iter__(self) -> Iterator[Stash]:
        return iter(self.stashes)

    def __getitem__(self, index: int) -> Stash:
        return self.stashes[index]

    def __len__(self) -> int:
        return len(self.stashes)

    def to_frame(
        self, stat: Callable[[Tensor], Tensor] = torch.std
    ) -> "pandas.DataFrame":  # type:ignore[name-defined] # NOQA: F821
        import pandas

        def to_item(stash: Stash) -> Dict[str, Any]:
            d = stash.__dict__.copy()
            first_value = stash.first_value
            d["value"] = (
                stat(first_value).item() if isinstance(first_value, Tensor) else None
            )
            d["type"] = f"{stash.type.__module__}.{stash.type.__name__}"
            return d

        return pandas.DataFrame.from_dict(map(to_item, self))  # type:ignore[arg-type]


def track(
    module: nn.Module,
    grad: bool = True,
    include: NamePattern = None,
    exclude: NamePattern = None,
    stash_value: Optional[StashValueFn] = None,
    stash: Optional[StashFn] = None,
) -> Tracker:
    tracker = Tracker(get_stash_fn(stash_value=stash_value, stash=stash))
    tracker.register_all(module, grad=grad, include=include, exclude=exclude)
    return tracker


track.__doc__ = __doc__

__all__ = [
    "Event",
    "Stash",
    "StashFn",
    "StashValueFn",
    "rmap_tensor",
    "default_stash_value",
    "default_stash",
    "get_stash_fn",
    "Tracker",
    "track",
]
