# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Basic utility for tracking activations and gradients at `nn.Module` outputs.

Basic usage:

```
with tensor_tracker.track(module) as tracker:
    module(inputs).backward()

print(list(tracker))
```

Advanced usage:

 - Filter modules based on name:
   `track(include="<regex>", exclude="<regex>")`

 - Pre-transform tracked tensors to save memory:
   `track(stash_value=lambda t: t.std().detach().cpu())`

 - Customise tracked state:
   `track(stash=lambda event: ...)`

 - Manually register/unregister hooks:
  `tracker = Tracker(); tracker.register(); tracker.unregister()`
"""

from .core import *  # NOQA: F401 F403
from .core import __all__  # NOQA: F401
