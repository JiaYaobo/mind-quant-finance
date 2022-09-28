from turtle import width
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mind_quant_finance.engine.nn.composed.mlp import MLP

class DriftField(nn.Cell):
    net: MLP
    in_size: int
    out_size: int
    width_size: int

    def __init__(self, in_size, out_size, width_size):
        super(DriftField, self).__init__()
        self._in_size = in_size
        self._out_size = out_size
        self._width_size = width_size
        self.net = MLP(in_size=in_size, out_size=out_size, width_size=width_size)

    def construct(self, x):
        return self.net(x)


class SigmaField(nn.Cell):
    net: MLP
    in_size: int
    out_size: int
    width_size: int
    
    def __init__(self, in_size, out_size, width_size):
        super(SigmaField, self).__init__()
        self._in_size = in_size
        self._out_size = out_size
        self._width_size = width_size
        self.net = MLP(in_size=in_size, out_size=out_size, width_size=width_size)

    def construct(self, x):
        return self.net(x)


class NeuralSDE(nn.Cell):
    
    def __init__(self):
        super(NeuralSDE, self).__init__()