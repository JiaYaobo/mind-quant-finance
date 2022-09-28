import time
import argparse
import mindspore
import mindspore.context as context
import mindspore.numpy as mnp
from mind_quant_finance.engine.nn.composed.mlp import MLP

def mlp_test():
    in_size = 3
    out_size = 4
    width_size = 5
    depth = 2
    x = mnp.randn((2, 3))
    mlp = MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth)
    return mlp(x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="black sholes options test")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        help="set which type of device you want to use. Ascend/GPU",
    )
    parser.add_argument(
        "--device_id", default=0, type=int, help="device id is for physical devices"
    )
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
    start = time.time()
    print(mlp_test())
    end = time.time()
    print(f"#1 time {end - start}")



