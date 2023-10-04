# Tensor tracker

[API documentation](https://graphcore-research.github.io/pytorch-tensor-tracker/) | [Example](https://graphcore-research.github.io/pytorch-tensor-tracker/usage.html)

Flexibly track outputs and grad-outputs of `torch.nn.Module`.

**Installation:**

```bash
pip install git+https://github.com/graphcore-research/pytorch-tensor-tracker
```

**Usage:**

Use `tensor_tracker.track(module)` as a context manager to start capturing tensors from within your module's forward and backward passes:

```python
import tensor_tracker

with tensor_tracker.track(module) as tracker:
    module(inputs).backward()

print(tracker)  # => Tracker(stashes=8, tracking=0)
```

Now `Tracker` is filled with stashes, containing copies of fwd/bwd tensors at (sub)module outputs. (Note, this can consume a lot of memory.)

It behaves like a list of `Stash` objects, with their attached `value`, usually a tensor or tuple of tensors. We can also use `to_frame()` to get a Pandas table of summary statistics:

```python
print(list(tracker))
# => [Stash(name="0.linear", type=nn.Linear, grad=False, value=tensor(...)),
#     ...]

display(tracker.to_frame())
```

<img src="doc/usage_to_frame.png" alt="tensor tracker to_frame output" style="width:30em;"/>

See the [documentation](https://graphcore-research.github.io/pytorch-tensor-tracker/) for more info, or for a more practical example, see our demo of [visualising transformer activations & gradients using UMAP](doc/Example.ipynb).


## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License ([LICENSE](LICENSE)).

Our dependencies are (see [requirements.txt](requirements.txt)):

| Component | About | License |
| --- | --- | --- |
| torch | Machine learning framework | BSD 3-Clause |

We also use additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).
