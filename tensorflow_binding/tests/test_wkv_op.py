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


def make_attention_mask(x):
    # Mask looks like (causual - diag):
    #
    # 0  0 ..   0
    # 1  0 ..   0
    # 1  1 ..   0
    # 1  1 ..   0
    #
    B, T, H = tf.shape(x)
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
    # sum_wkv = tf.einsum('tuh,buh->bth', exp_w, exp_kv)                # [B;U;H]
    # sum_wk = tf.einsum('tuh,buh->bth', exp_w, exp_k)                  # [B;U;H]
    #
    exp_w = tf.transpose(exp_w, perm=[2, 1, 0])                         # [H;U;T]
    exp_k = tf.transpose(exp_k, perm=[2, 0, 1])                         # [H;B;U]
    exp_kv = tf.transpose(exp_kv, perm=[2, 0, 1])                       # [H;B;U]
    sum_wk = tf.transpose(tf.matmul(exp_k, exp_w), perm=[1, 2, 0])      # [B;T;H]
    sum_wkv = tf.transpose(tf.matmul(exp_kv, exp_w), perm=[1, 2, 0])    # [B;T;H]
    wkv = (sum_wkv + exp_bk * v) / (sum_wk + exp_bk)                    # [B;U;H]
    return wkv

# ===============================================================================================
#                                       Tests

class WKVTest(tf.test.TestCase):

    @staticmethod
    def _make_inputs():
        B, T, C = 20, 30, 40
        return tf.random.normal((C,)), tf.random.normal((C,)), tf.random.normal((B, T, C,)), tf.random.normal((B, T, C,))

    def test_wkv_forward(self):
        w, u, k, v = self._make_inputs()

        base = compute_wkv_base(w, u, k, v)
        new = WKV(w, u, k, v)
        self.assertAllClose(base, new, rtol=1e-3, atol=1e-3)

    def test_wkv_backward(self):
        w, u, k, v = self._make_inputs()
        vars_ = [w, u, k, v]

        # Compute base grads.
        #
        with tf.GradientTape() as tape:
            for v in vars_:
                tape.watch(v)
            y = compute_wkv_base(w, u, k, v)

        base_grads = tape.gradient(y, vars_)

        # Compute new grads.
        #
        with tf.GradientTape() as tape:
            for v in vars_:
                tape.watch(v)
            y = WKV(w, u, k, v)

        new_grads = tape.gradient(y, vars_)
        self.assertAllClose(base_grads, new_grads, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    set_random()
    tf.test.main()
