import numpy as np
import random
from rwkv_cuda import WKV
import tensorflow as tf

# ===============================================================================================
#                                       Helpers

def set_random():
    random.seed(2)
    tf.random.set_seed(2)
    np.random.seed(2)


def shape_list(x, out_type=tf.int32):
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def make_attention_mask(x):
    # Mask looks like (causual - diag):
    #
    # 0  0 ..   0
    # 1  0 ..   0
    # 1  1 ..   0
    # 1  1 ..   0
    #
    B, T, H = shape_list(x)
    ones = tf.ones(shape=(T, T))
    lower = tf.linalg.band_part(ones, -1, 0)
    diag = tf.linalg.band_part(ones, 0, 0)
    return lower - diag


def compute_wkv_base(w, u, k, v):
    # w equals to -exp(log_decay)
    # u equals to bonus

    mask = tf.expand_dims(make_attention_mask(v), axis=-1)              # [T;U;1]
    weights = tf.cumsum(mask, exclusive=True, axis=0)                   # [T;U;1]

    w = weights * w                                                     # [T;U;H]
    k = tf.clip_by_value(k, -60, 30)                                    # [B;U;H]
    exp_w = tf.math.exp(w) * mask                                       # [T;U;H]
    exp_k = tf.math.exp(k)                                              # [B;U;H]
    exp_kv = exp_k * v                                                  # [B;U;H]
    exp_b = tf.math.exp(u)                                              # [H]
    exp_bk = exp_b * exp_k                                              # [B;U;H]

    # The same as (but slightly faster):
    #
    #sum_wkv = tf.einsum('tuh,buh->bth', exp_w, exp_kv)                # [B;U;H]
    #sum_wk = tf.einsum('tuh,buh->bth', exp_w, exp_k)                  # [B;U;H]
    #
    exp_w = tf.transpose(exp_w, perm=[2, 1, 0])                         # [H;U;T]
    exp_k = tf.transpose(exp_k, perm=[2, 0, 1])                         # [H;B;U]
    exp_kv = tf.transpose(exp_kv, perm=[2, 0, 1])                       # [H;B;U]
    sum_wk = tf.transpose(tf.matmul(exp_k, exp_w), perm=[1, 2, 0])      # [B;T;H]
    sum_wkv = tf.transpose(tf.matmul(exp_kv, exp_w), perm=[1, 2, 0])    # [B;T;H]
    wkv = (sum_wkv + exp_bk * v) / (sum_wk + exp_bk)                    # [B;U;H]
    return wkv

# ===============================================================================================
#                                       Benchmark

def _make_inputs():
    B, T, C = 2 * 1024 * 10, 24, 300
    return tf.random.normal((C,)), tf.random.normal((C,)), tf.random.normal((B, T, C,)), tf.random.normal((B, T, C,))


def benchmark(op):
    with tf.Graph().as_default() as graph:
        w, u, k, v = _make_inputs()
        out = op(w, u, k, v)

        with tf.compat.v1.Session(graph=graph) as sess:
            bench = tf.test.Benchmark()
            ret = bench.run_op_benchmark(
                sess,
                out,
                min_iters=100,
                store_memory_usage=False,
            )
        return ret

def print_bench(name, bench):
    print(f'===== {name}')
    time = bench['extras']['wall_time_mean'] * 1000.
    stddev = bench['extras']['wall_time_stdev'] * 1000.
    print(f'time = {time:.2f} +/- {stddev:.2f}ms')
    print()


# Typical benchmark tests on v100.
#
# Base (keras implementation):
#   einsum:                     135ms
#   trannspose + matmul:         90ms
#
# Kernel:
#   no_optim:                    60ms
#  + extra-device-vectorization  58ms
#  + use_fast_math               58ms
#
if __name__ == '__main__':
    set_random()

    base_bench = benchmark(compute_wkv_base)
    print_bench('base', base_bench)

    kernel_bench = benchmark(WKV)
    print_bench('kernel', kernel_bench)
