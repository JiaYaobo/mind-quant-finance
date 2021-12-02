import numpy as np
from mindspore import Tensor
import mindspore.numpy as mnp
import time
from mindspore import ms_function
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU",
                    device_id=0, save_graphs=False, enable_graph_kernel=True)


def init_mnp():
    x = mnp.arange(1024).reshape(2, 512).astype('float32')
    w1 = mnp.ones((512, 1024))
    b1 = mnp.zeros((1024,))
    w2 = mnp.ones((1024, 2048))
    b2 = mnp.zeros((2048,))
    w3 = mnp.ones((2048, 4096))
    b3 = mnp.zeros((4096,))

    return x, w1, b1, w2, b2, w3, b3


def init_np():
    x = np.arange(1024).reshape(2, 512).astype('float32')
    w1 = np.ones((512, 1024))
    b1 = np.zeros((1024,))
    w2 = np.ones((1024, 2048))
    b2 = np.zeros((2048,))
    w3 = np.ones((2048, 4096))
    b3 = np.zeros((4096,))

    return x, w1, b1, w2, b2, w3, b3


def forward_np(x, w1, b1, w2, b2, w3, b3):
    x = np.dot(x, w1) + b1
    x = np.dot(x, w2) + b2
    x = np.dot(x, w3) + b3
    return x


def forward(x, w1, b1, w2, b2, w3, b3):
    x = mnp.dot(x, w1) + b1
    x = mnp.dot(x, w2) + b2
    x = mnp.dot(x, w3) + b3
    return x


@ms_function
def forward_compiled(x, w1, b1, w2, b2, w3, b3):
    x = mnp.dot(x, w1) + b1
    x = mnp.dot(x, w2) + b2
    x = mnp.dot(x, w3) + b3

    return x


x, w1, b1, w2, b2, w3, b3 = init_mnp()

t = forward(x, w1, b1, w2, b2, w3, b3)
start = time.time()
t = forward(x, w1, b1, w2, b2, w3, b3)
end = time.time()
print(f"mindpore without compiled: {end - start}")


tc = forward_compiled(x, w1, b1, w2, b2, w3, b3)
start = time.time()
tc = forward_compiled(x, w1, b1, w2, b2, w3, b3)
end = time.time()
print(f"mindspore with graph compiled: {end - start}")

x, w1, b1, w2, b2, w3, b3 = init_np()
tc = forward_np(x, w1, b1, w2, b2, w3, b3)
start = time.time()
tc = forward_np(x, w1, b1, w2, b2, w3, b3)
end = time.time()
print(f"numpy: {end - start}")
