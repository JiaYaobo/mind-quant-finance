import tensorflow as tf
import time

def normal():
    dtype=tf.float32
    
    x = tf.random.normal(shape=[10000, 20],
                               dtype=dtype,
                               mean=0.0, stddev=1.0,
                               seed=1234)
    samples = tf.matmul(x, tf.transpose(x))

    return samples

@tf.function
def xla_normal():
    dtype=tf.float32
    
    x = tf.random.normal(shape=[10000, 20],
                               dtype=dtype,
                               mean=0.0, stddev=1.0,
                               seed=1234)
    samples = tf.matmul(x, tf.transpose(x))

    return samples


if __name__ == '__main__':
    start_ts = time.time()
    s = normal()
    # print(s[:1])
    end_ts = time.time()
    print(f"noxla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = normal()
    s.numpy()
    end_ts = time.time()
    print(f"noxla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = normal()
    s.numpy()
    end_ts = time.time()
    print(f"noxla #2 {end_ts - start_ts}")
    
    start_ts = time.time()
    s = xla_normal()
    s.numpy()
    end_ts = time.time()
    print(f"xla #1 {end_ts - start_ts}")

    start_ts = time.time()
    s = xla_normal()
    s.numpy()
    end_ts = time.time()
    print(f"xla #2 {end_ts - start_ts}")