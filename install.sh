#!/usr/bin/env bash
set -euo pipefail

# RWKV: Build cuda kernels.
#
export RWKV_CUDA_PATH=$(pwd)/build
export CUDA_HOME=/usr/local/cuda-11.1
mkdir build && cd build
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME .. && make

# RWKV: install python library.
#
cd ../tensorflow_binding
pip3 install . --use-feature=in-tree-build

# RWKV: test & cleanup
#
cd ../..
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export CUDA_VISIBLE_DEVICES=0
python3 -m site
python3 rwkv-cuda/tensorflow_binding/tests/test_wkv_op.py

