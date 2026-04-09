#!/bin/bash
set -e
source /home/zyjin/anaconda3/etc/profile.d/conda.sh
conda activate dp_devel
export DP_VARIANT=cuda

cd source
mkdir -p build && cd build
PT_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
cmake -DENABLE_PYTORCH=ON -DCMAKE_PREFIX_PATH=${PT_PATH} -DUSE_TF_BACKEND=OFF -DUSE_PT_BACKEND=ON -DCMAKE_CXX_STANDARD=17 -DPYTHON_EXECUTABLE=$(which python) -DOP_CXX_ABI_PT=0 -DCMAKE_INSTALL_PREFIX=../.. -DCMAKE_CUDA_ARCHITECTURES="120" -DTORCH_CUDA_ARCH_LIST='12.0' ..
make -j4 deepmd_op_pt
