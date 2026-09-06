/**
 * smem_txn_demo.cu — 共享内存 bank conflict 按 128B transaction 计，不是按整个 warp 计
 *
 * 截图观点：
 *   一次 shared 事务固定 128B（32 bank × 4B）。
 *   每线程 8B/16B 时，一个 warp 的请求会被拆成多个 128B transaction，
 *   conflict 只在同一个 transaction 内算。
 *   因此 float2 连续访问：T0 和 T16 同 bank，但分属两次事务 → 0 conflict。
 *
 * 对照实验（全部 1 block × 32 thread，显式 PTX，避免 ptxas 拆成标量 LDS）：
 *   kF32_seq      LDS.32  连续     32×4B  = 128B → 1 wavefront, 0 conflict
 *   kF64_seq      LDS.64  连续     32×8B  = 256B → 2 wavefronts, 0 conflict  ← 截图例子
 *   kF128_seq     LDS.128 连续     32×16B = 512B → 4 wavefronts, 0 conflict
 *   kF32_2way     LDS.32  stride=2 同 128B 内两线程打同一 bank → 真 2-way conflict
 *   kF64_stride2  LDS.64  stride=2 T0 与 T8 同 bank、跨 128B 行；看硬件怎么打包
 *   kF32_bcast    LDS.32  全 warp 同地址 → broadcast，0 conflict
 *
 * 截图里的 C++ `float2 val = smem[threadIdx.x]` 会编成 LDS.64；ptxas 容易把循环
 * 不变地址提出去，所以这里用 `ld.volatile.shared.v2.b32` 钉住同一访问图案。
 *
 * 编译:
 *   nvcc -O3 -lineinfo -arch=sm_120 -o /tmp/smem_txn profile/study/smem_txn_demo.cu
 *
 * ncu（本文件 main 只 launch 一次每个 kernel）:
 *   ncu --csv --metrics \
 *     smsp__inst_executed_op_shared_ld.sum,\
 *     l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
 *     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
 *     smsp__sass_data_bytes_mem_shared_op_ld.sum,\
 *     smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ld \
 *     /tmp/smem_txn
 *
 * RTX 5050 Laptop (sm_120) 实测，每条 LDS 归一化：
 *                 wf/LDS  conflict/LDS  B/wavefront
 *   kF32_seq         1.0           0          128     连续 LDS.32，1 次事务
 *   kF64_seq         2.0           0          128     截图例子：T0/T16 同 bank 但分两次事务
 *   kF128_seq        4.0           0          128     float4 同理，拆 4 次
 *   kF32_2way        2.0           1           64     真 2-way：128B 请求打成 2 次事务
 *   kF64_stride2     4.0           2           64     向量化也会 conflict，若同事务内撞 bank
 *   kF32_bcast       1.0           0            4     广播免费，不算 conflict
 *
 * 单 warp 的 cyc/inst 是延迟，看不出 conflict 惩罚；以 ncu 的 wf/conflict 为准。
 */

#include <cstdio>
#include <cuda_runtime.h>

#ifndef ITERS
#define ITERS 50000
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

// 8 条独立地址链，间距 512B，保证不同 128B 行，避免跨 iter 别名干扰。
static constexpr int kStrideReg = 512;
static constexpr int kSmemBytes = UNROLL * kStrideReg;

__device__ __forceinline__ void lds32(unsigned& x, unsigned addr) {
  asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(x) : "r"(addr));
}
__device__ __forceinline__ void lds64(unsigned& x, unsigned& y, unsigned addr) {
  asm volatile("ld.volatile.shared.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
}
__device__ __forceinline__ void lds128(unsigned& x, unsigned& y, unsigned& z, unsigned& w,
                                       unsigned addr) {
  asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "r"(addr));
}

__device__ __forceinline__ long long clk() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ long long clk_dep(unsigned dep) {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) : "r"(dep));
  return t;
}

// W = 4/8/16 字节；stride_b = 每线程相对 lane0 的字节步长；0 = 广播。
template <int W, int STRIDE_B>
__device__ void load_loop(float* out, long long* tk, int iters) {
  extern __shared__ __align__(16) char smem[];
  const unsigned lane = threadIdx.x & 31u;
  for (int i = (int)lane; i < kSmemBytes / 4; i += 32) {
    reinterpret_cast<float*>(smem)[i] = (float)i;
  }
  __syncwarp();

  unsigned base = (unsigned)__cvta_generic_to_shared(smem);
  unsigned q[UNROLL][4];
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    q[u][0] = q[u][1] = q[u][2] = q[u][3] = 0;
  }

  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int j = 0; j < iters; ++j) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      unsigned off = (STRIDE_B == 0) ? 0u : lane * (unsigned)STRIDE_B;
      unsigned addr = base + (unsigned)u * kStrideReg + off;
      if (W == 4) {
        lds32(q[u][0], addr);
      } else if (W == 8) {
        lds64(q[u][0], q[u][1], addr);
      } else {
        lds128(q[u][0], q[u][1], q[u][2], q[u][3], addr);
      }
    }
  }
  unsigned sink = 0;
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) {
    sink += q[u][0] + q[u][1] + q[u][2] + q[u][3];
  }
  long long t1 = clk_dep(sink);
  out[threadIdx.x] = (float)sink;
  if (threadIdx.x == 0) *tk = t1 - t0;
}

extern "C" __global__ void kF32_seq(float* o, long long* t, int n) { load_loop<4, 4>(o, t, n); }
extern "C" __global__ void kF64_seq(float* o, long long* t, int n) { load_loop<8, 8>(o, t, n); }
extern "C" __global__ void kF128_seq(float* o, long long* t, int n) { load_loop<16, 16>(o, t, n); }
extern "C" __global__ void kF32_2way(float* o, long long* t, int n) { load_loop<4, 8>(o, t, n); }
extern "C" __global__ void kF64_stride2(float* o, long long* t, int n) { load_loop<8, 16>(o, t, n); }
extern "C" __global__ void kF32_bcast(float* o, long long* t, int n) { load_loop<4, 0>(o, t, n); }

using Kptr = void (*)(float*, long long*, int);

void run(const char* name, Kptr k, int smem, int loads_per_iter) {
  float* o;
  long long* t;
  cudaMalloc(&o, 32 * 4);
  cudaMalloc(&t, 8);
  if (smem > 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  // 单次 launch：方便 ncu 每个 kernel 只出现一行。clock64 在 device 侧，不需要 host warmup。
  k<<<1, 32, smem>>>(o, t, ITERS);
  cudaDeviceSynchronize();
  long long cyc = 0;
  cudaMemcpy(&cyc, t, 8, cudaMemcpyDeviceToHost);
  double inst = (double)ITERS * loads_per_iter;
  printf("%-14s  cyc/inst=%7.2f  (loop=%d x loads/iter=%d, 1 warp)\n", name, (double)cyc / inst,
         ITERS, loads_per_iter);
  cudaFree(o);
  cudaFree(t);
}

int main() {
  printf("ITERS=%d UNROLL=%d smem=%dB  arch: see nvcc -arch\n", ITERS, UNROLL, kSmemBytes);
  run("kF32_seq", kF32_seq, kSmemBytes, UNROLL);
  run("kF64_seq", kF64_seq, kSmemBytes, UNROLL);
  run("kF128_seq", kF128_seq, kSmemBytes, UNROLL);
  run("kF32_2way", kF32_2way, kSmemBytes, UNROLL);
  run("kF64_stride2", kF64_stride2, kSmemBytes, UNROLL);
  run("kF32_bcast", kF32_bcast, kSmemBytes, UNROLL);
  return 0;
}
