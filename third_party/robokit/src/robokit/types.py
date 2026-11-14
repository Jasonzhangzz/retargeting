from typing import TYPE_CHECKING, Union

import numpy as np


if TYPE_CHECKING:
    import torch
    import warp as wp

ArrayLike = Union[np.ndarray, "torch.Tensor", "wp.array"]
