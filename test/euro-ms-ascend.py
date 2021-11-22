import argparse
import time

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

import mindspore.ops as ops
from mindspore import context

class EuroCallPrice(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, m, spot, volatility, strike, maturity):
        super(EuroCallPrice, self).__init__()
        # 定义所需要的运算
        self.m = m
        self.spot = spot
        self.volatility = volatility
        self.strike = strike
        self.maturity = maturity

    def construct(self, WTs):
        # 使用定义好的运算构建前向网络
        drift = 0.5 * self.volatility ** 2 * self.maturity
        # WTs = ops.normal((self.m,), Tensor(0.0, mindspore.float32), std)
        STs = ops.Mul()(Tensor(self.spot, mindspore.float32), ops.Exp()(ops.Add()(Tensor(-drift, mindspore.float32), ops.Mul()(Tensor(self.volatility, mindspore.float32), WTs))))
        payoffs = ops.Maximum()(Tensor(0.0, mindspore.float32), ops.Add()(STs, Tensor(-self.strike, mindspore.float32)))
        result = ops.ReduceMean(keep_dims=True)(payoffs)
        
        return result


if __name__ == '__main__':
    m = 1000000
    spot = 276.10
    volatility = 0.407530933
    strike = 230
    maturity = 58 / 365
    std = np.sqrt(maturity)
    context.set_context(device_target="Ascend", device_id=7)
    x = np.random.normal(0.0, std, (m, ))
    start_ts = time.time()
    model = EuroCallPrice(m, spot, volatility, strike, maturity)
    result = model(x)
    print(result)
    first = time.time()
	# r = estimate_call_price_ms(m, spot, volatility, strike, maturity)
    result = model(x)

    print(time.time() - start_ts)