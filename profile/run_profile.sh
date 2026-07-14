#!/bin/bash
set -e
cd /root/learn-cuda-cute-triton/profile
CUDA=/usr/local/cuda
NVCC=$CUDA/bin/nvcc
NCU=$CUDA/bin/ncu
ARCH=sm_90
echo "NVCC=$NVCC NCU=$NCU ARCH=$ARCH"

echo "=== [1/3] roofline_demo ==="
$NVCC -std=c++17 -O3 -arch=$ARCH -o roofline_demo roofline_demo.cu
$NCU --set full --launch-skip 3 --launch-count 3 -f -o roofline_demo ./roofline_demo

echo "=== [2/3] stall_demo ==="
$NVCC -std=c++17 -O3 -arch=$ARCH -o stall_demo stall_demo.cu
$NCU --set full --launch-skip 0 --launch-count 4 -f -o stall_demo ./stall_demo

echo "=== [3/3] stream_demo ==="
$NVCC -std=c++17 -O3 -arch=$ARCH -o stream_demo stream_demo.cu
$NCU --set full --launch-skip 1 --launch-count 6 -f -o stream_demo ./stream_demo

echo "==== DONE ===="
ls -la *.ncu-rep
