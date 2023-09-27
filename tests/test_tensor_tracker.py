# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

import tensor_tracker

pytestmark = pytest.mark.filterwarnings("ignore:.+backward hooks:UserWarning")


class Mul3(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * 3


@dataclass
class Output:
    thing: Tensor
    status: str


class EgModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m3 = Mul3()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Output:
        return Output(thing=self.sigmoid(self.m3(x=x)), status="ok")


def test_basic() -> None:
    module = EgModule()
    with tensor_tracker.track(module) as tracker:
        x = torch.full((8,), 0.7, requires_grad=True)
        out = module(x)
        grad = torch.full_like(out.thing, 1000)
        out.thing.backward(grad)

    sigmoid_grad = 1000 * torch.sigmoid(3 * x) * (1 - torch.sigmoid(3 * x))
    expected_stash = [
        tensor_tracker.Stash("m3", Mul3, False, 3 * x),
        tensor_tracker.Stash("sigmoid", nn.Sigmoid, False, torch.sigmoid(3 * x)),
        tensor_tracker.Stash("", EgModule, False, Output(torch.sigmoid(3 * x), "ok")),
        # tensor_tracker.Stash("", EgModule, True, grad),
        tensor_tracker.Stash("sigmoid", nn.Sigmoid, True, grad),
        tensor_tracker.Stash("m3", Mul3, True, sigmoid_grad),
    ]
    assert len(tracker) == len(expected_stash)
    for i, expected in enumerate(expected_stash):
        assert tracker[i].name == expected.name
        assert tracker[i].type == expected.type
        assert tracker[i].grad == expected.grad
        if isinstance(expected.value, Output):
            assert isinstance(tracker[i].value, Output)
            assert torch.equal(tracker[i].value.thing, expected.value.thing)
            assert tracker[i].value.status == expected.value.status
        else:
            assert torch.equal(tracker[i].first_value, expected.value), expected

    df = tracker.to_frame()
    assert len(df) == len(expected_stash)
    df.iloc[0] == pd.Series(
        dict(name="m3", type="test_tensor_tracker.Mul3", grad=False, value=0.0)
    )


def test_custom_stash_value() -> None:
    torch.manual_seed(100)
    module = EgModule()
    with tensor_tracker.track(
        module, stash_value=lambda t: t.std().cpu().detach()
    ) as tracker:
        x = torch.randn(1000)
        module(x)

    assert tracker[0].name == "m3"
    assert tracker[0].value == (3 * x).std()
    assert all(
        s.first_value.ndim == 0 for s in tracker if isinstance(s.first_value, Tensor)
    )


def test_custom_stash() -> None:
    def custom_stash(event: tensor_tracker.Event) -> tensor_tracker.Stash:
        stash = tensor_tracker.Stash(
            event.name,
            event.type,
            event.grad,
            tensor_tracker.rmap_tensor(event.value, tensor_tracker.default_stash_value),
        )
        args = tensor_tracker.rmap_tensor(
            event.args, tensor_tracker.default_stash_value
        )
        setattr(stash, "args", args)
        kwargs = tensor_tracker.rmap_tensor(
            event.kwargs, tensor_tracker.default_stash_value
        )
        setattr(stash, "kwargs", kwargs)
        return stash

    module = EgModule()
    with tensor_tracker.track(module, stash=custom_stash) as tracker:
        x = torch.ones(1000)
        module(x)

    assert tracker[0].name == "m3"
    assert torch.equal(tracker[0].value, 3 * x)
    assert torch.equal(tracker[0].kwargs["x"], x)  # type:ignore[attr-defined]
    assert torch.equal(tracker[1].args[0], 3 * x)  # type:ignore[attr-defined]

    # Cannot specify both stash=? and stash_value=?
    with pytest.raises(ValueError), tensor_tracker.track(
        module, stash=custom_stash, stash_value=lambda t: t
    ):
        pass
