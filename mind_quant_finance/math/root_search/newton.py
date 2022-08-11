from mindspore import dtype as mstype
import mindspore as ms
import mindspore.numpy as np
import numpy as onp
from mindspore import ms_function
from typing import Optional, Callable


def newton_root_finder(val_grad_func: Callable,
                       r0: ms.Tensor,
                       max_iterations: Optional[int] = 20,
                       tolerance: Optional[float] = 1e-6,
                       relative_tolerance: Optional[float] = None
                       ):
    """Args:
        val_grad_func: A Python Callback Function, return tuple(val,grad)
                    (1) val: f(r), the value of the function,
                    (2) grad: D(f(r))/D(r), the grad of the function.
                    while val and grad have the same shape/dtype with r0.
        r0: The initial values of the parameter to use.
            (the starting point of Newton Method)
        max_iterations: The maximum number of iterations of Newton's method.
        tolerance: |f(x_n)-f(x_n-1)|<|x_n|*relative_tolerance + tolerance
        relative_tolerance: positive double, default float.
    """
    """Return: 
        r: the solution given by Newton Root Solver.
        converged: indicating whether the corresponding root results in an objective
                   function value less than the tolerance.
        failed: indicating whether the corresponding 'root' is not finite.
        root not converged & not failed: limit by the max_iterations, 
                    not fail but also not converged, need to increase the max_iterations
                    to get the needy tolerance / accuracy
    """
    if not relative_tolerance:
        relative_tolerance = ms.Tensor(4 * onp.finfo(float).eps)
    count = 0
    r = r0
    converged = np.zeros_like(r0, dtype=mstype.bool_)
    failed = np.zeros_like(r0, dtype=mstype.bool_)
    while count < max_iterations:
        val, grad = val_grad_func(r)
        count, r, converged, failed, _need_stop = \
            _newton_one_step(count, val, grad, converged, r, failed,
                             relative_tolerance, tolerance)
        if _need_stop:
            break
    return r, converged, failed


@ms_function
def _newton_one_step(count, val, grad, converged, r, failed, relative_tolerance, tolerance):
    delta = np.divide(val, grad)
    converged = np.logical_or(converged,
                              (np.abs(delta) < relative_tolerance * np.abs(val) + tolerance))
    update_mask = (~np.logical_or(converged, failed)).astype(r.dtype)
    r = r - update_mask * delta
    failed = np.logical_or(failed, ~np.isfinite(r))
    _need_stop = (np.logical_or(converged, failed)).all()
    return count + 1, r, converged, failed, _need_stop
