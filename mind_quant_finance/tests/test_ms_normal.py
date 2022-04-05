import argparse
import time

import numpy as np
import mindspore
from mindspore import Tensor

import mindspore.ops as P
from mindspore import context
from mindspore import ms_function

@ms_function
def normal(shape, mean, stddev):
    stdnormal = P.StandardNormal(2, 3)
    stdnormal.add_prim_attr("use_curand", True)
    random_normal = stdnormal(shape)
    value = random_normal * stddev + mean
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test normal performance')
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='set which type of device you want to use. Ascend/GPU')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id is for physical devices')

    args = parser.parse_args()

    context.set_context(device_target=args.device_target,
                        device_id=args.device_id, mode=context.GRAPH_MODE,
                    save_graphs=False, enable_graph_kernel=True)

    start_ts = time.time()
    mean = Tensor(0.0, mindspore.float32)
    stddev = Tensor(1.0, mindspore.float32)
    normal((10000000, 2), mean, stddev)
    end_ts = time.time()

    print(f"#1 {end_ts - start_ts}")

    start_ts = time.time()
    normal((10000000, 2), mean, stddev)
    end_ts = time.time()

    print(f"#2 {end_ts - start_ts}")

    # start_ts = time.time()
    # normal((100000, 2), mean, stddev)
    # end_ts = time.time()

    # print(f"#3 {end_ts - start_ts}")



    