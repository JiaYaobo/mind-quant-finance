import numpy as np
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor


def uniform_grid(minimums: list,
                 maximums: list,
                 sizes: list,
                 dtype: mindspore.dtype = mindspore.float32,
                 validate_args: bool = False) -> Tensor:
    if not all(x > 2 for x in sizes):
        raise ValueError(
            'The sizes of the grid must be greater than 1.')

    if not (len(minimums) == len(maximums) and len(minimums) == len(sizes)):
        raise ValueError(
            'The shapes of minimums, maximums and sizes must be identical.')

    if not any(isinstance(i, list) for i in minimums):
        raise ValueError(
            'The minimums, maximums and sizes must all be rank 1.')

    locations = [
        mnp.linspace(minimums[i], maximums[i], num=sizes[i]) for i in range(len(sizes))
    ]

    return locations
