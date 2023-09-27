# Tensor tracker

Flexibly track outputs and grad-outputs of `torch.nn.Module`.

```python
with tensor_tracker.track(module) as tracker:
    module(inputs).backward()

print(list(tracker))

display(tracker.to_frame())

print(tensor_tracker.__doc__)  # further usage
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License ([LICENSE](LICENSE)).

Our dependencies are (see [requirements.txt](requirements.txt)):

| Component | About | License |
| --- | --- | --- |
| torch | Machine learning framework | BSD 3-Clause |

We also use additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).
