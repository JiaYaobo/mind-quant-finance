# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
create piece wise function given data and jump locations
"""

import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mindspore import ms_function

__all__ = [
    "PiecewiseConstantFunction"
]


@ms_function
def _piecewise_constant_function(x, jump_locations, values, side='left'):
    """piece wise const function"""
    indices = mnp.searchsorted(jump_locations, x, side=side)
    axis = 0
    res = P.Gather()(values, indices, axis)
    return res


class PiecewiseConstantFunction(nn.Cell):

    def __init__(self, jump_locations, values, dtype=None):
        """
        Piecewise Constant Function

        Args:
            x (mindspore.Tensor): input of the piecewise function
            jump_locations (mindspore.Tensor): jump locations of piecewise function
            values(mindspore.Tensor): values to 
            side (str): indicates which side is continuous

        Returns:
            mindspre.Tensor, function values on given x
            
        Examples:
            >>> from mind_quant_finance.math import piecewise 
            >>> x = mnp.array([0., 0.1, 2. ,11.])
            >>> jump_locations = mnp.array([0.1, 10], dtype=dtype)
            >>> values = mnp.array([3, 4, 5], dtype=dtype)
            >>> piecewise_func = piecewise.PiecewiseConstantFunction(jump_locations, values, dtype=dtype)
            >>> computed_value = piecewise_func(x)
            >>> print(computed_value)
            [3., 3., 4., 5.]
        """

        super(PiecewiseConstantFunction, self).__init__()

        self._jump_locations = mnp.array(jump_locations, dtype=dtype)
        self._dtype = dtype or self._jump_locations
        self._values = mnp.array(values, dtype=self._dtype)
        self._piecewise_constant_function = _piecewise_constant_function

    def construct(self, x):
        return self._piecewise_constant_function(x, self._jump_locations, self._values)
