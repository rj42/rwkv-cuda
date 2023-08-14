import imp
import sys
import tensorflow as tf
from tensorflow.python.framework import ops

if not tf.config.list_physical_devices('GPU'):
    raise ImportError(f'Failed to load {__name__} library: no cuda devices.')
else:
    print(f'Successfully loaded {__name__} library!', file=sys.stderr)


lib_file = imp.find_module('kernels', __path__)[1]
_rwkv_cuda = tf.load_op_library(lib_file)


def WKV(w, u, k, v):
    return _rwkv_cuda.wkv_forward(w, u, k, v)

@ops.RegisterGradient("WKV_FORWARD")
def _wkv_backward(op, grad):
    w, u, k, v = op.inputs
    y = op.outputs[0]
    gy = grad
    gw, gu, gk, gv = _rwkv_cuda.wkv_backward(w, u, k, v, y, gy)
    gw = tf.reduce_sum(gw, axis=0)
    gu = tf.reduce_sum(gu, axis=0)
    return (gw, gu, gk, gv)
