import time
import argparse

import numpy as np
import mindspore
import mindspore.ops as P
import mindspore.context as context
import mindspore.numpy as mnp
from mindspore import Tensor

import times


def test_grid_from_time_step():
    dtype = mindspore.float32
    tolerance = Tensor(1e-6, dtype)
    times_array = [0.1, 0.5, 1, 2]
    time_step = 0.1

    all_times, time_indices = times._grid_from_time_step(times=times_array,
                                                         time_step=time_step,
                                                         dtype=dtype,
                                                         tolerance=tolerance)
    print("test_grid_from_time_step")
    print(f"all_times: {all_times}")
    print(f"time_indices: {time_indices}")


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

    test_grid_from_time_step()