import mindspore.numpy as np
import mindspore as ms
from mindspore import ms_function, ops

SQRT_2 = np.sqrt(ms.Tensor(2.0))
RSQRT_2 = 1.0 / SQRT_2
SQRT_2PI = np.sqrt(2.0 * np.pi)
RSQRT_2PI = 1.0 / SQRT_2PI


@ms_function
def standnormal_pdf(x):
    return np.exp(-0.5 * (x ** 2)) * RSQRT_2PI


@ms_function
def standnormal_cdf(x):
    x2 = x * RSQRT_2
    return 0.5 * (1.0 + ops.erf(x2))
