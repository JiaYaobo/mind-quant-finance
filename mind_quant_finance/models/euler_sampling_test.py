import os

os.environ['GLOG_v'] = '3'
import time
import argparse

import numpy as np
import mindspore
import mindspore.ops as P
import mindspore.context as context
import mindspore.numpy as mnp
from mindspore import Tensor

from random_sampler import RandomType

import euler_sampling


def test_sample_paths_1d(use_batch, supply_normal_draws, random_type):
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
    dtype = mindspore.float16
    mu = 0.2
    a = 0.4
    b = 0.33

    def drift_fn(t, x):
        drift = mu * P.Sqrt()(t) * P.Ones()(x.shape, t.dtype)
        return drift

    def vol_fn(t, x):
        return (a * t + b) * P.Ones()((1, 1), t.dtype)

    times = 0.55
    num_samples = 10000

    normal_draws = None

    if use_batch:
        # x0.shape = [2, 1, 1]
        x0 = np.array([[[0.1]], [[0.1]]])
    else:
        x0 = np.array([0.1])
    paths = euler_sampling.sample(dim=1,
                                  drift_fn=drift_fn,
                                  volatility_fn=vol_fn,
                                  times=times,
                                  num_samples=num_samples,
                                  initial_state=x0,
                                  random_type=random_type,
                                  normal_draws=normal_draws,
                                  time_step=0.01,
                                  seed=42,
                                  dtype=dtype)
    paths = paths.asnumpy()

    means = np.mean(paths, axis=-2)
    expected_means = x0 + (2.0 / 3.0) * mu * np.power(times, 1.5)
    print(f"means {means}")
    print(f"expected_means {expected_means}")
    np.testing.assert_array_almost_equal(means, expected_means, decimal=2)


def test_sample_paths_2d(random_type):
    """Tests path properties for 2-dimentional Ito process."""
    # We construct the following Ito processes.
    # dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2
    # dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2
    # mu_1, mu_2 are constants.
    # s_ij = a_ij t + b_ij
    # For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.
    dtype = mindspore.float32
    num_samples = 1000000
    times = 1
    time_step = 1
    x0 = np.array([0.1, -1.1])
    mu = np.array([0.2, 0.7])
    a = np.array([[0.4, 0.1], [0.3, 0.2]])
    b = np.array([[0.33, -0.03], [0.21, 0.5]])

    mu = Tensor(mu, dtype)
    a = Tensor(a, dtype)
    b = Tensor(b, dtype)

    def drift_fn(t, x):
        return mu * P.Sqrt()(t) * P.Ones()(x.shape, t.dtype)

    def vol_fn(t, x):
        del x
        return (a * t + b) * P.Ones()((2, 2), t.dtype)

    print(f"test_sample_paths_2d {random_type}")

    start = time.time()
    paths = euler_sampling.sample(dim=2,
                                  drift_fn=drift_fn,
                                  volatility_fn=vol_fn,
                                  times=times,
                                  num_samples=num_samples,
                                  initial_state=x0,
                                  random_type=random_type,
                                  time_step=time_step,
                                  seed=1,
                                  dtype=dtype)
    end = time.time()
    print(f"#1 time: {end - start}")

    start = time.time()
    paths = euler_sampling.sample(dim=2,
                                  drift_fn=drift_fn,
                                  volatility_fn=vol_fn,
                                  times=times,
                                  num_samples=num_samples,
                                  initial_state=x0,
                                  random_type=random_type,
                                  time_step=time_step,
                                  seed=1,
                                  dtype=dtype)
    end = time.time()
    print(f"#2 time: {end - start}")
    paths = paths.asnumpy()
    print("test_halton_sample_paths_2d")
    # print(f"paths {paths}")
    means = np.mean(paths, axis=0)
    print(f"paths mean {means}")

    # self.assertAllClose(paths.shape, (num_samples, 3, 2), atol=0)
    # means = np.mean(paths, axis=0)
    # times = np.reshape(times, [-1, 1])

    expected_means = x0 + (2.0 / 3.0) * mu.asnumpy() * np.power(times, 1.5)

    print(f"expected_means: {expected_means}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="euler sampling test")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        help="set which type of device you want to use. Ascend/GPU",
    )
    parser.add_argument("--device_id",
                        default=0,
                        type=int,
                        help="device id is for physical devices")
    parser.add_argument(
        "--enable_graph_kernel",
        default=True,
        type=bool,
        help="whether to use graph kernel",
    )
    args = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
        save_graphs=False,
        enable_graph_kernel=args.enable_graph_kernel,
    )

    test_sample_paths_1d(use_batch=False,
                         supply_normal_draws=False,
                         random_type=RandomType.PSEUDO_ANTITHETIC)

    test_sample_paths_2d(random_type=RandomType.PSEUDO)

    test_sample_paths_2d(random_type=RandomType.SOBOL)
