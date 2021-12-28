import tensorflow as tf
import time

def normal():
    dtype=tf.float32
    
    samples = tf.random.normal(shape=[10000000, 2],
                               dtype=dtype,
                               mean=0.0, stddev=1.0,
                               seed=1234)
    return samples

@tf.function
def xla_normal():
    dtype=tf.float32
    
    samples = tf.random.normal(shape=[10000000, 2],
                               dtype=dtype,
                               mean=0.0, stddev=1.0,
                               seed=1234)
    return samples


if __name__ == '__main__':
    start_ts = time.time()
    s = normal()
    end_ts = time.time()
    print(f"noxla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = normal()
    end_ts = time.time()
    print(f"noxla #2 {end_ts - start_ts}")
    
    start_ts = time.time()
    xla_normal()
    end_ts = time.time()
    print(f"xla #1 {end_ts - start_ts}")

    start_ts = time.time()
    xla_normal()
    end_ts = time.time()
    print(f"xla #2 {end_ts - start_ts}")