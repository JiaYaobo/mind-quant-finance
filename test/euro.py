import argparse
import time

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as ds

import mindspore.ops as ops
from mindspore import context

class EuroCallPrice(nn.Cell):
    """
    网络结构
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


def get_data(iter_num, data_size, mean, std):
    for _ in range(iter_num):
        data = np.random.normal(mean, std, (data_size, ))
        yield (data.astype(np.float32), )


def create_dataset(iter_num, data_size, mean, std, repeat_size=1, batch_size=1):
    """定义数据集"""
    input_data = ds.GeneratorDataset(list(get_data(iter_num, data_size, mean, std)), column_names=['data'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='euro call options')
    parser.add_argument('--device_target', type=str, default='Ascend', help='set which type of device you want to use. Ascend/GPU')
    parser.add_argument('--device_id', default=1, type=int, help='device id is for physical devices')

    args = parser.parse_args()

    m = 1000000
    spot = 276.10
    volatility = 0.407530933
    strike = 230
    maturity = 58 / 365
    std = np.sqrt(maturity)
    context.set_context(device_target=args.device_target, device_id=args.device_id)
    x = np.random.normal(0.0, std, (m, ))

    model = EuroCallPrice(m, spot, volatility, strike, maturity)
    syn_data = create_dataset(3, m, 0.0, std)

    for x in syn_data:
        start_ts = time.time()
        result = model(x[0])
        print(result)
        print(time.time() - start_ts)