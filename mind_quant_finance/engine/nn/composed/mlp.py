from mindspore.nn.layer import Dense, CellList
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as P
from typing import Callable, List
from mindspore import ms_function



@ms_function
def _identity(x: Tensor):
    return x 

class MLP(nn.Cell):
    layers: CellList
    activation: Callable
    final_activation: Callable
    depth: int
    in_size: int
    out_size: int
    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = nn.ReLU(),
        final_activation: Callable = _identity,
        has_bias=False
    ):

        super().__init__()
        layers = CellList()
        if depth == 0:
            layers.append(Dense(in_size, out_size, has_bias=has_bias))
        else:
            layers.append(Dense(in_size, width_size, has_bias=has_bias))
            for i in range(depth - 1):
                layers.append(Dense(width_size, width_size, has_bias=has_bias))
            layers.append(Dense(width_size, out_size, has_bias=has_bias))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
    

    def construct(self, x: Tensor):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x

    