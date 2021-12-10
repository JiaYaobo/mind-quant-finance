import mindspore
import mindspore.ops as P
from mindspore import Tensor


def matvec(a: Tensor, b: Tensor):
    """Multiplies matrix `a` by vector `b`, producing `a` * `b`.
    `b` can also be a matrix.
    """
    return P.Squeeze(-1)(P.matmul(a, P.ExpandDims()(b, -1)))