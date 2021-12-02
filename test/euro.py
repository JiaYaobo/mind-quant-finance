import argparse
import time

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.ops.primitive import constexpr

import mindspore.ops as ops
from mindspore import context


@constexpr
def _init_params(m, spot, volatility, maturity, strike, dtype):
    std = np.sqrt(maturity)
    WTs = ops.normal((m, ), Tensor(0.0, dtype), Tensor(std, dtype), seed=5)
    spot_tensor = Tensor(spot, dtype)
    negative_drift = (-0.5) * volatility ** 2 * maturity
    negative_drift_tensor = Tensor(negative_drift, dtype)
    volatility_tensor = Tensor(volatility, dtype)
    negative_strike_tensor = Tensor((-strike), dtype)
    zero_tensor = Tensor(0.0, dtype)
    return WTs, spot_tensor, negative_drift_tensor, volatility_tensor, negative_strike_tensor, zero_tensor


class EuroCallPrice(nn.Cell):
    """
    网络结构
    """

    def __init__(self, m, spot, volatility, strike, maturity, dtype):
        super(EuroCallPrice, self).__init__()
        # 定义所需要的运算
        self.m = m
        self.spot = spot
        self.volatility = volatility
        self.strike = strike
        self.maturity = maturity
        self.dtype = dtype

    def construct(self):
        # WTs = ops.normal((self.m,), Tensor(0.0, mindspore.float32), std)
        WTs, spot, neg_drift, volatility, neg_strike, zero = _init_params(
            self.m, self.spot, self.volatility, self.maturity, self.strike, self.dtype)
        STs = ops.Mul()(spot, ops.Exp()(ops.Add()(neg_drift, ops.Mul()
                                                  (volatility, WTs))))
        payoffs = ops.Maximum()(zero, ops.Add()
                                (STs, neg_strike))
        result = ops.ReduceMean(keep_dims=True)(payoffs)

        return result


def get_data(iter_num, data_size, mean, std):
    for _ in range(iter_num):
        data = np.random.normal(mean, std, (data_size, ))
        yield (data.astype(np.float32), )


def create_dataset(iter_num, data_size, mean, std, repeat_size=1, batch_size=1):
    """定义数据集"""
    input_data = ds.GeneratorDataset(
        list(get_data(iter_num, data_size, mean, std)), column_names=['data'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='euro call options')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='set which type of device you want to use. Ascend/GPU')
    parser.add_argument('--device_id', default=1, type=int,
                        help='device id is for physical devices')

    args = parser.parse_args()

    m = 1000000
    spot = 276.10
    volatility = 0.407530933
    strike = 230
    maturity = 58 / 365
    std = np.sqrt(maturity)
    context.set_context(device_target=args.device_target,
                        device_id=args.device_id)
    # x = np.random.normal(0.0, std, (m, ))

    model = EuroCallPrice(m, spot, volatility, strike,
                          maturity, mindspore.float32)
    # syn_data = create_dataset(3, m, 0.0, std)

    for x in range(3):
        start_ts = time.time()
        result = model()
        print(result.item())
        print(time.time() - start_ts)
