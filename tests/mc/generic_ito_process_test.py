import os

os.environ['GLOG_v'] = '3'
import time
import argparse

import mindspore
import mindspore.ops as P
import mindspore.context as context
import mindspore.numpy as np
from mindspore import Tensor

from mind_quant_finance.mc.generic_ito_process import GenericltoProcess


def test_sample_paths_1d(use_batch):
    """Tests path properties for 1-dimentional Ito process.

    We construct the following Ito process.

    ````
    dX = mu * sqrt(t) * dt + (a * t + b) dW
    ````

    For this process expected value at time t is x_0 + 2/3 * mu * t^1.5 .
    Args:
      use_batch: Test parameter to specify if we are testing the batch of Euler
        sampling.
      supply_normal_draws: Supply normal draws.
      random_type: `RandomType` of the sampled normal draws.
    """

    dtype = mindspore.float32
    mu = 0.2
    a = 0.4
    b = 0.33
    dim = 1

    def drift_fn(t, x):
        drift = mu * P.Sqrt()(t) * P.Ones()(x.shape, t.dtype)
        return drift

    def vol_fn(t, x):
        return (a * t + b)

        # return (a * t + b) * P.Ones()((1, 10000, 1), t.dtype)

    process = GenericltoProcess(1, drift_fn, vol_fn)
    # times = 0.55
    num_paths = 10000
    T = 1.0
    num_timesteps = 55
    dt = T / num_timesteps
    normal_draws = np.rand(1, num_paths, num_timesteps, dim)

    if use_batch:
        # x0.shape = [2, 1, 1]
        x0 = np.array([[[0.1]], [[0.1]]])
    else:
        x0 = np.array([0.1])
    
    start = time.time()
    paths = process(x0, 1, num_paths, num_timesteps, dt, normal_draws)
    print(f"# 1: {time.time() - start}")
    # paths = paths.asnumpy()
    # print(paths)

    start = time.time()
    paths = process(x0, 1, num_paths, num_timesteps, dt, normal_draws)
    print(f"# 1: {time.time() - start}")


context.set_context(
        mode=context.GRAPH_MODE,
        device_target="GPU",
        device_id=0,
        save_graphs=False
)

test_sample_paths_1d(use_batch=False)



    # means = np.mean(paths, axis=-2)
    # expected_means = x0 + (2.0 / 3.0) * mu * np.power(T, 1.5)
    # print(f"means {means}")
    # print(f"expected_means {expected_means}")
    # np.testing.assert_array_almost_equal(means, expected_means, decimal=2)