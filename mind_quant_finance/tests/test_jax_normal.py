import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import device_put
import time

def normal():
    dtype=jnp.float32

    key = random.PRNGKey(1234)
    x = random.normal(key=key, 
        shape=[10000, 20],
        dtype=dtype)

    samples = jnp.matmul(x, x.T).block_until_ready()
    
    return samples

@jit
def xla_normal():
    dtype=jnp.float32

    key = random.PRNGKey(1234)
    x = random.normal(key=key, 
        shape=[10000, 20],
        dtype=dtype)

    samples = jnp.matmul(x, x.T)
    
    return samples


if __name__ == '__main__':
    start_ts = time.time()
    s = normal()
    np.asarray(s)
    end_ts = time.time()
    print(f"noxla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = normal()
    np.asarray(s)
    end_ts = time.time()
    print(f"noxla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = normal()
    np.asarray(s)
    end_ts = time.time()
    print(f"noxla #2 {end_ts - start_ts}")

    start_ts = time.time()
    s = xla_normal()
    np.asarray(s)
    end_ts = time.time()
    print(f"xla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = xla_normal()
    np.asarray(s)
    end_ts = time.time()
    print(f"xla #2 {end_ts - start_ts}")