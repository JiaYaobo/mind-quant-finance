"Generic Ito Process"

from typing import Callable

import mindspore
import mindspore.ops as P
import mindspore.numpy as np
from mindspore import Tensor

import mindspore.nn as nn

from mind_quant_finance.math.ops import matvec


class GenericltoProcess(nn.Cell):
    def __init__(self, 
        dim: int, 
        dirft_fn: Callable, 
        volatility_fn: Callable):
        super(GenericltoProcess, self).__init__()
        self.dim = dim
        self.drift_fn = dirft_fn
        self.volatility_fn = volatility_fn

    def construct(self, initial_state, batch_size, num_paths, num_timesteps, dt, normal_draws):
        i = P.ScalarToTensor()(0, mindspore.int32)
        num_timesteps = Tensor(num_timesteps, mindspore.int32)
        current_state = np.broadcast_to(initial_state, (batch_size, num_paths, self.dim))

        while i < num_timesteps:
            # t1=self.getint(t)
            dw = normal_draws[:,:,i,:]
            current_time = i * dt
            dt_inc = self.drift_fn(current_time,current_state) * dt
            dw_inc = self.volatility_fn(current_time,current_state) * dw
            next_state = current_state + dt_inc + dw_inc
            current_state = next_state

            i += 1
        return current_state