// =============================================================================
// LSU vs TMA peak bandwidth, and whether they add when used together
// Target: RTX 5050 Laptop / sm_120
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -lcuda -o lsu_tma_bw lsu_tma_bw.cu
//   ./lsu_tma_bw
//
// 2026-08-22 measured (20 SM, 32 MB L2, 128-bit GDDR7):
//   DRAM uni: LSU rd 365, TMA rd 357, LSU wr 367, TMA wr 368 GB/s
//   DRAM copy R+W: 335 GB/s  (not ~730 — bus is shared, not 2x duplex)
//   DRAM LSU||TMA two reads: aggregate 337-364 GB/s = same roof, not 2x
//   L2  : LSU 512, TMA 482; LSU||TMA 649 GB/s  (paths partially add)
//   L1  : LSU reuse 4611 GB/s; TMA cannot use L1
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult e = (call);                                                       \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *s = nullptr;                                                 \
      cuGetErrorString(e, &s);                                                 \
      fprintf(stderr, "CU %s:%d: %s\n", __FILE__, __LINE__, s);                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 128;
static constexpr int THREADS = 256;
static constexpr int TILE_BYTES = TILE_M * TILE_N * (int)sizeof(float);

__device__ __forceinline__ uint32_t cvta_to_shared(const void *p) {
  uint32_t a;
  asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
               : "=r"(a)
               : "l"(p));
  return a;
}
__device__ __forceinline__ void mbar_init(uint64_t *b, uint32_t cnt) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(cvta_to_shared(b)),
               "r"(cnt)
               : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t *b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                   cvta_to_shared(b)),
               "r"(tx)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *b, uint32_t phase) {
  asm volatile("{\n.reg .pred p;\n"
               "LOOP: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
               "  @!p bra LOOP;}\n" ::"r"(cvta_to_shared(b)),
               "r"(phase)
               : "memory");
}
__device__ __forceinline__ void tma_load_2d(void *dst, uint64_t desc, int cx,
                                            int cy, uint64_t *bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_"
               "tx::bytes [%0], [%1, {%3, %4}], [%2];\n" ::"r"(cvta_to_shared(dst)),
               "l"(desc), "r"(cvta_to_shared(bar)), "r"(cx), "r"(cy)
               : "memory");
}
__device__ __forceinline__ void tma_store_2d(void *src, uint64_t desc, int cx,
                                             int cy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n" ::
          "l"(desc),
      "r"(cvta_to_shared(src)), "r"(cx), "r"(cy)
      : "memory");
}
__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// =============================================================================
// LSU kernels
// =============================================================================
__global__ void lsu_read(const float4 *__restrict__ g, int n4, float *sink) {
  int stride = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    float4 v = g[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  if (acc.x + acc.y + acc.z + acc.w == 1e30f)
    *sink = acc.x;
}

__global__ void lsu_write(float4 *__restrict__ g, int n4) {
  int stride = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  for (; i < n4; i += stride)
    g[i] = v;
}

__global__ void lsu_copy(const float4 *__restrict__ s, float4 *__restrict__ d,
                         int n4) {
  int stride = gridDim.x * blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n4; i += stride)
    d[i] = s[i];
}

// Repeatedly touch a tiny tile so hits stay in L1 (LSU) or L2 (TMA, no L1).
__global__ void lsu_read_reuse(const float4 *__restrict__ g, int n4, int iters,
                               float *sink) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (int t = 0; t < iters; ++t) {
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      float4 v = g[i];
      acc.x += v.x;
      acc.y += v.y;
    }
  }
  if (acc.x + acc.y == 1e30f)
    *sink = acc.x;
}

// =============================================================================
// TMA kernels — 2D tensor [M, TILE_N], each block one 64x128 tile
// =============================================================================
__global__ void __launch_bounds__(THREADS) tma_read(const __grid_constant__ CUtensorMap tma,
                                                    float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  int n0 = 0;
  int m0 = blockIdx.x * TILE_M;
  if (threadIdx.x == 0)
    mbar_init(bar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect_tx(bar, TILE_BYTES);
    tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma), n0, m0, bar);
  }
  mbar_wait(bar, 0);
  if (threadIdx.x == 0 && smem[0] == 1e30f)
    *sink = smem[0];
}

__global__ void __launch_bounds__(THREADS)
    tma_read_reuse(const __grid_constant__ CUtensorMap tma, int iters, float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  if (threadIdx.x == 0)
    mbar_init(bar, 1);
  __syncthreads();
  // Always the same tile (row 0). TMA bypasses L1 → this is L2/DRAM reuse.
  for (int t = 0; t < iters; ++t) {
    uint32_t phase = t & 1;
    if (threadIdx.x == 0) {
      mbar_expect_tx(bar, TILE_BYTES);
      tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma), 0, 0, bar);
    }
    mbar_wait(bar, phase);
  }
  if (threadIdx.x == 0 && smem[0] == 1e30f)
    *sink = smem[0];
}

__global__ void __launch_bounds__(THREADS)
    tma_write(const __grid_constant__ CUtensorMap tma) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  for (int i = threadIdx.x; i < TILE_M * TILE_N; i += blockDim.x)
    smem[i] = (float)i;
  __syncthreads();
  tma_store_fence();
  int m0 = blockIdx.x * TILE_M;
  if (threadIdx.x == 0) {
    tma_store_2d(smem, reinterpret_cast<uint64_t>(&tma), 0, m0);
    tma_store_commit();
  }
  tma_store_wait();
}

// Warp-specialized: warp0 TMA-reads A, other warps LSU-read B. Concurrent on SM.
__global__ void __launch_bounds__(THREADS)
    dual_tma_read_lsu_read(const __grid_constant__ CUtensorMap tma_a,
                           const float4 *__restrict__ b, int n4_b, int n_tiles_a,
                           float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  float acc = 0.f;

  if (warp == 0) {
    if (lane == 0)
      mbar_init(bar, 1);
    __syncwarp();
    // Strip of tiles along M, this block owns blockIdx.x, then steps by grid.
    for (int tile = blockIdx.x; tile < n_tiles_a; tile += gridDim.x) {
      uint32_t phase = (tile / gridDim.x) & 1;
      if (lane == 0) {
        mbar_expect_tx(bar, TILE_BYTES);
        tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma_a), 0, tile * TILE_M, bar);
      }
      mbar_wait(bar, phase);
      acc += smem[lane];
    }
  } else {
    int nwarps_lsu = 7;
    int lid = (warp - 1) * 32 + lane;
    int stride = gridDim.x * nwarps_lsu * 32;
    int i = blockIdx.x * nwarps_lsu * 32 + lid;
    for (; i < n4_b; i += stride) {
      float4 v = b[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 1e30f)
    *sink = acc;
}

// Warp-spec: warp0 TMA-reads A, other warps LSU-write C (duplex).
__global__ void __launch_bounds__(THREADS)
    dual_tma_read_lsu_write(const __grid_constant__ CUtensorMap tma_a,
                            float4 *__restrict__ c, int n4_c, int n_tiles_a,
                            float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  if (warp == 0) {
    if (lane == 0)
      mbar_init(bar, 1);
    __syncwarp();
    for (int tile = blockIdx.x; tile < n_tiles_a; tile += gridDim.x) {
      uint32_t phase = (tile / gridDim.x) & 1;
      if (lane == 0) {
        mbar_expect_tx(bar, TILE_BYTES);
        tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma_a), 0, tile * TILE_M, bar);
      }
      mbar_wait(bar, phase);
    }
    if (lane == 0 && smem[0] == 1e30f)
      *sink = smem[0];
  } else {
    int nwarps_lsu = 7;
    int lid = (warp - 1) * 32 + lane;
    int stride = gridDim.x * nwarps_lsu * 32;
    int i = blockIdx.x * nwarps_lsu * 32 + lid;
    float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (; i < n4_c; i += stride)
      c[i] = v;
  }
}

// Warp-spec: warp0 TMA-writes A (from smem pattern), other warps LSU-read B.
__global__ void __launch_bounds__(THREADS)
    dual_tma_write_lsu_read(const __grid_constant__ CUtensorMap tma_a,
                            const float4 *__restrict__ b, int n4_b, int n_tiles_a,
                            float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < TILE_M * TILE_N; i += blockDim.x)
    smem[i] = (float)i;
  __syncthreads();
  float acc = 0.f;
  if (warp == 0) {
    tma_store_fence();
    for (int tile = blockIdx.x; tile < n_tiles_a; tile += gridDim.x) {
      if (lane == 0) {
        tma_store_2d(smem, reinterpret_cast<uint64_t>(&tma_a), 0, tile * TILE_M);
        tma_store_commit();
      }
      tma_store_wait();
    }
  } else {
    int nwarps_lsu = 7;
    int lid = (warp - 1) * 32 + lane;
    int stride = gridDim.x * nwarps_lsu * 32;
    int i = blockIdx.x * nwarps_lsu * 32 + lid;
    for (; i < n4_b; i += stride) {
      float4 v = b[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 1e30f)
    *sink = acc;
}

// =============================================================================
// Host
// =============================================================================
static CUtensorMap make_tma(void *p, int M, int N) {
  CUtensorMap d;
  cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)M};
  cuuint64_t str[1] = {(cuuint64_t)N * sizeof(float)};
  cuuint32_t box[2] = {(cuuint32_t)TILE_N, (cuuint32_t)TILE_M};
  cuuint32_t es[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, p, dims, str, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return d;
}

struct Stats {
  float med, mean, stdev;
};

template <typename Launch>
static Stats time_stats(Launch &&launch, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> v(reps);
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&v[i], a, b));
  }
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  std::sort(v.begin(), v.end());
  float mean = std::accumulate(v.begin(), v.end(), 0.f) / reps;
  float var = 0.f;
  for (float x : v)
    var += (x - mean) * (x - mean);
  return {v[reps / 2], mean, sqrtf(var / reps)};
}

static void show(const char *name, double bytes, Stats s) {
  double gbs = bytes / 1e9 / (s.med / 1e3);
  printf("  %-32s  med=%7.3f ms  %7.1f GB/s\n", name, s.med, gbs);
}

int main() {
  CU_CHECK(cuInit(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int SM = prop.multiProcessorCount;
  printf("================================================================\n");
  printf(" LSU vs TMA bandwidth   %s  sm_%d%d  SMs=%d  L2=%d MB  bus=%d-bit\n",
         prop.name, prop.major, prop.minor, SM, prop.l2CacheSize / 1024 / 1024,
         prop.memoryBusWidth);
  printf("================================================================\n");

  int smem = TILE_BYTES + 16;
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_read,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_read_reuse,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_write,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)dual_tma_read_lsu_read,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)dual_tma_read_lsu_write,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)dual_tma_write_lsu_read,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  float *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, 4));

  auto lsu_grid = [&]() {
    return dim3(SM * 6); // 6*256 = 1536 = max threads/SM
  };

  // ----- 1. DRAM unidirectional + copy -----
  {
    // 262144 rows * 128 cols * 4B = 128 MB per buffer
    const int Md = 262144;
    const int N = TILE_N;
    const int n_tiles = Md / TILE_M;
    const int n4 = (int)((size_t)Md * N / 4);
    size_t bytes = (size_t)Md * N * sizeof(float);
    printf("\n=== 1. DRAM  (one buffer = %.1f MB, no reuse) ===\n", bytes / 1e6);

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 1, bytes));
    CUDA_CHECK(cudaMemset(B, 1, bytes));
    CUtensorMap tA = make_tma(A, Md, N);
    CUtensorMap tC = make_tma(C, Md, N);

    dim3 lg = lsu_grid();
    dim3 tg(n_tiles);
    int warm = 5, reps = 20;

    show("LSU read", (double)bytes,
         time_stats([&] { lsu_read<<<lg, THREADS>>>((float4 *)A, n4, sink); }, warm,
                    reps));
    show("TMA read", (double)bytes,
         time_stats([&] { tma_read<<<tg, THREADS, smem>>>(tA, sink); }, warm, reps));
    show("LSU write", (double)bytes,
         time_stats([&] { lsu_write<<<lg, THREADS>>>((float4 *)C, n4); }, warm,
                    reps));
    show("TMA write", (double)bytes,
         time_stats([&] { tma_write<<<tg, THREADS, smem>>>(tC); }, warm, reps));
    show("LSU copy  (R+W)", 2.0 * bytes,
         time_stats([&] { lsu_copy<<<lg, THREADS>>>((float4 *)A, (float4 *)C, n4); },
                    warm, reps));

    // Two-stream concurrent kernels, occupancy-capped so they can coreside.
    {
      int cap = 2; // blocks/SM each
      dim3 g1(SM * cap), g2(n_tiles);
      // TMA grid is n_tiles which is huge; limit TMA to SM*cap as well by
      // launching persistent-ish: still n_tiles blocks is many waves.
      // For stream overlap use same large grids — GPU will time-share if full.
      cudaStream_t s1, s2;
      CUDA_CHECK(cudaStreamCreate(&s1));
      CUDA_CHECK(cudaStreamCreate(&s2));

      auto both_read = [&] {
        lsu_read<<<g1, THREADS, 0, s1>>>((float4 *)A, n4, sink);
        tma_read<<<g2, THREADS, smem, s2>>>(tA, sink);
        CUDA_CHECK(cudaStreamSynchronize(s1));
        CUDA_CHECK(cudaStreamSynchronize(s2));
      };
      // Sequential baseline: same two launches one stream
      auto seq_read = [&] {
        lsu_read<<<g1, THREADS, 0, s1>>>((float4 *)A, n4, sink);
        tma_read<<<g2, THREADS, smem, s1>>>(tA, sink);
        CUDA_CHECK(cudaStreamSynchronize(s1));
      };
      printf("\n  -- two kernels, each reads the SAME %.1f MB --\n", bytes / 1e6);
      show("seq  LSU-read then TMA-read", 2.0 * bytes,
           time_stats(seq_read, warm, reps));
      show("par  LSU-read || TMA-read  ", 2.0 * bytes,
           time_stats(both_read, warm, reps));

      auto both_duplex = [&] {
        tma_read<<<g2, THREADS, smem, s1>>>(tA, sink);
        lsu_write<<<g1, THREADS, 0, s2>>>((float4 *)C, n4);
        CUDA_CHECK(cudaStreamSynchronize(s1));
        CUDA_CHECK(cudaStreamSynchronize(s2));
      };
      auto seq_duplex = [&] {
        tma_read<<<g2, THREADS, smem, s1>>>(tA, sink);
        lsu_write<<<g1, THREADS, 0, s1>>>((float4 *)C, n4);
        CUDA_CHECK(cudaStreamSynchronize(s1));
      };
      printf("  -- duplex: TMA-read A || LSU-write C --\n");
      show("seq  TMA-read then LSU-write", 2.0 * bytes,
           time_stats(seq_duplex, warm, reps));
      show("par  TMA-read || LSU-write  ", 2.0 * bytes,
           time_stats(both_duplex, warm, reps));

      CUDA_CHECK(cudaStreamDestroy(s1));
      CUDA_CHECK(cudaStreamDestroy(s2));
    }

    // Same-CTA warp specialization (guaranteed concurrent issue)
    printf("\n  -- same CTA, warp0=TMA, warp1-7=LSU --\n");
    dim3 dg(SM * 2);
    show("dual TMA-read + LSU-read", 2.0 * bytes,
         time_stats(
             [&] {
               dual_tma_read_lsu_read<<<dg, THREADS, smem>>>(tA, (float4 *)B, n4,
                                                             n_tiles, sink);
             },
             warm, reps));
    show("dual TMA-read + LSU-write", 2.0 * bytes,
         time_stats(
             [&] {
               dual_tma_read_lsu_write<<<dg, THREADS, smem>>>(tA, (float4 *)C, n4,
                                                              n_tiles, sink);
             },
             warm, reps));
    show("dual TMA-write + LSU-read", 2.0 * bytes,
         time_stats(
             [&] {
               dual_tma_write_lsu_read<<<dg, THREADS, smem>>>(tC, (float4 *)B, n4,
                                                              n_tiles, sink);
             },
             warm, reps));

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }

  // ----- 2. L2-resident (paths might add if L2 ports are independent) -----
  {
    // 4 MB per buffer, two buffers = 8 MB << 32 MB L2
    const int Md = 8192; // 8192*128*4 = 4 MB
    const int N = TILE_N;
    const int n_tiles = Md / TILE_M;
    const int n4 = (int)((size_t)Md * N / 4);
    size_t bytes = (size_t)Md * N * sizeof(float);
    const int kIters = 64; // repeat to make it measurable; still L2
    printf("\n=== 2. L2-resident  (%.2f MB x2, %d iters) ===\n", bytes / 1e6, kIters);

    float *A = nullptr, *B = nullptr;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMemset(A, 1, bytes));
    CUDA_CHECK(cudaMemset(B, 1, bytes));
    CUtensorMap tA = make_tma(A, Md, N);
    dim3 lg = lsu_grid();
    dim3 tg(n_tiles);
    int warm = 10, reps = 30;

    auto lsu_rep = [&] {
      for (int i = 0; i < kIters; ++i)
        lsu_read<<<lg, THREADS>>>((float4 *)A, n4, sink);
    };
    auto tma_rep = [&] {
      for (int i = 0; i < kIters; ++i)
        tma_read<<<tg, THREADS, smem>>>(tA, sink);
    };
    show("LSU read  (L2)", (double)bytes * kIters, time_stats(lsu_rep, warm, reps));
    show("TMA read  (L2)", (double)bytes * kIters, time_stats(tma_rep, warm, reps));

    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));
    auto par = [&] {
      for (int i = 0; i < kIters; ++i) {
        lsu_read<<<lg, THREADS, 0, s1>>>((float4 *)B, n4, sink);
        tma_read<<<tg, THREADS, smem, s2>>>(tA, sink);
      }
      CUDA_CHECK(cudaStreamSynchronize(s1));
      CUDA_CHECK(cudaStreamSynchronize(s2));
    };
    auto seq = [&] {
      for (int i = 0; i < kIters; ++i) {
        lsu_read<<<lg, THREADS, 0, s1>>>((float4 *)B, n4, sink);
        tma_read<<<tg, THREADS, smem, s1>>>(tA, sink);
      }
      CUDA_CHECK(cudaStreamSynchronize(s1));
    };
    show("seq LSU+TMA  (2 arrays, L2)", 2.0 * bytes * kIters,
         time_stats(seq, warm, reps));
    show("par LSU||TMA (2 arrays, L2)", 2.0 * bytes * kIters,
         time_stats(par, warm, reps));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    cudaFree(A);
    cudaFree(B);
  }

  // ----- 3. Tiny reuse: L1 for LSU, L2 for TMA -----
  {
    const int n4_tile = 256; // 4 KB
    const int iters = 4096;
    printf("\n=== 3. Tiny reuse  (4 KB tile x %d iters). LSU→L1, TMA→L2 ===\n",
           iters);
    float4 *g = nullptr;
    CUDA_CHECK(cudaMalloc(&g, n4_tile * 16));
    CUDA_CHECK(cudaMemset(g, 1, n4_tile * 16));
    // TMA reuse kernel uses a 64x128 = 32 KB box; allocate that.
    float *big = nullptr;
    CUDA_CHECK(cudaMalloc(&big, TILE_BYTES));
    CUDA_CHECK(cudaMemset(big, 1, TILE_BYTES));
    CUtensorMap t = make_tma(big, TILE_M, TILE_N);

    double lsu_bytes = (double)SM * 4 * n4_tile * 16 * iters; // 4 blocks/SM
    show("LSU read reuse (expect L1)", lsu_bytes,
         time_stats(
             [&] { lsu_read_reuse<<<SM * 4, THREADS>>>(g, n4_tile, iters, sink); },
             10, 30));

    double tma_bytes = (double)TILE_BYTES * iters; // 1 CTA hammering one tile
    show("TMA read reuse x1 CTA (expect L2)", tma_bytes,
         time_stats(
             [&] { tma_read_reuse<<<1, THREADS, smem>>>(t, iters, sink); }, 10, 30));
    double tma_bytes_all = (double)TILE_BYTES * iters * SM;
    show("TMA read reuse xN SM (expect L2)", tma_bytes_all,
         time_stats(
             [&] { tma_read_reuse<<<SM, THREADS, smem>>>(t, iters, sink); }, 10,
             30));
    cudaFree(g);
    cudaFree(big);
  }

  printf("\n================================================================\n");
  printf("done.\n");
  return 0;
}
