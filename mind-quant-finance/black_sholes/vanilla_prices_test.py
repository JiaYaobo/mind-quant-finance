import time
import argparse
import mindspore
import mindspore.context as context
import mindspore.numpy as mnp

import vanilla_prices


def test_option_prices():
    """Tests that the BS prices are correct."""
    forwards = mnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    strikes = mnp.array([3.0, 3.0, 3.0, 3.0, 3.0])
    volatilities = mnp.array([0.0001, 102.0, 2.0, 0.1, 0.4])
    expiries = 1.0
    computed_prices = vanilla_prices.option_price(
        volatilities=volatilities,
        strikes=strikes,
        expiries=expiries,
        forwards=forwards,
        dtype=mindspore.float32)
    expected_prices = mnp.array(
        [0.0, 2.0, 2.0480684764112578, 1.0002029716043364, 2.0730313058959933])

    isclose = mnp.isclose(computed_prices, expected_prices, 1e-6)
    print(f"is correct {isclose}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='black sholes options test')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='set which type of device you want to use. Ascend/GPU')
    parser.add_argument('--device_id', default=1, type=int,
                        help='device id is for physical devices')
    parser.add_argument('--enable_graph_kernel', default=True, type=bool,
                        help='whether to use graph kernel')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        device_id=args.device_id, save_graphs=False, enable_graph_kernel=args.enable_graph_kernel)
    start = time.time()
    test_option_prices()
    end = time.time()
    print(f"#1 time {end - start}")

    start = time.time()
    test_option_prices()
    end = time.time()
    print(f"#2 time {end - start}")
