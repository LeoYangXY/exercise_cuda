// =============================================================================
// Mask: TMA vs software prefetch  (Blackwell sm_120 / RTX 5050 Laptop)
//
// Operator:  dst[m, n] = (n < mask[m]) ? src[m, n] : 0
//            src, dst : [M, N] row-major fp32
//            mask     : [M] int32, 0..N
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_120 -lcuda \
//        -o mask_tma_vs_prefetch mask_tma_vs_prefetch.cu
//
// Run:
//   ./mask_tma_vs_prefetch           # all experiments
//   ./mask_tma_vs_prefetch conc      # TMA outstanding K=1,2,3
//   ./mask_tma_vs_prefetch check     # correctness vs scalar ref
//   ./mask_tma_vs_prefetch hbm       # median GB/s, 4096^2 and 8192^2
//   ./mask_tma_vs_prefetch size|tiny|pat
//   ./mask_tma_vs_prefetch prof      # 3 kernels x5, for ncu
//
// Measured (RTX 5050 Laptop, 20 SM, 32 MB L2, 128-bit GDDR7, 2026-08-22):
//   HBM 8192^2 (WS 537 MB): ALL coalesced kernels sit at 336-340 GB/s.
//     copy_vec4 339, vec4_2d 340, tma_one 337, tma_pipe2 339, row_swp4 338
//     strided 1-thread-per-row: 207 GB/s  (the original scheme-2 mapping)
//   ncu 4096^2: vec4 and TMA both ~88% DRAM peak; TMA L1-LSU load sectors
//     16k vs vec4 2.2M (TMA bypasses L1); strided 65% DRAM peak, 4.2M sectors.
//   TMA concurrency, 1 CTA, 32 KB tiles: K=2 1.09x, K=3 1.20x vs serial.
//   Conclusion: this op is HBM-bound. TMA is not 20-50% faster than a
//     coalesced float4 kernel. Prefetch distance does not help. The original
//     1-thread-per-row "prefetch" loses because it is stride-N, not because
//     it lacks a TMA unit.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <tuple>
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

// =============================================================================
// PTX helpers (same sequence as origin_cuda_kernel/sgemm/hgemm_tma_warp_spec.cu)
// =============================================================================
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
  asm volatile("{\n"
               "  .reg .pred p;\n"
               "LOOP: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
               "  @!p bra LOOP;\n"
               "}\n" ::"r"(cvta_to_shared(b)),
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

__device__ __forceinline__ void cp_async_16B(void *smem, const void *gmem) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                   cvta_to_shared(smem)),
               "l"(gmem)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
  // n must be a literal for PTX; callers only pass 0 or 1
  if (n == 0)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  else
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

__device__ __forceinline__ void prefetch_l2(const void *p) {
  asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p));
}

// Apply mask for one 64x128 tile sitting in smem, STG as float4.
__device__ __forceinline__ void apply_tile_stg(const float *smem, float *dst,
                                               const int *smask, int m0, int n0,
                                               int N) {
  constexpr int kVec = (TILE_M * TILE_N) / THREADS / 4; // 8
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    int e4 = threadIdx.x + i * THREADS;
    int elem = e4 * 4;
    int row = elem / TILE_N;
    int col = elem % TILE_N;
    float4 v = *reinterpret_cast<const float4 *>(smem + elem);
    int mval = smask[row];
    int gc = n0 + col;
    float4 o;
    o.x = (gc + 0 < mval) ? v.x : 0.f;
    o.y = (gc + 1 < mval) ? v.y : 0.f;
    o.z = (gc + 2 < mval) ? v.z : 0.f;
    o.w = (gc + 3 < mval) ? v.w : 0.f;
    *reinterpret_cast<float4 *>(dst + (size_t)(m0 + row) * N + gc) = o;
  }
}

__device__ __forceinline__ void apply_tile_inplace(float *smem, const int *smask,
                                                   int n0) {
  constexpr int kVec = (TILE_M * TILE_N) / THREADS / 4;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    int e4 = threadIdx.x + i * THREADS;
    int elem = e4 * 4;
    int row = elem / TILE_N;
    int col = elem % TILE_N;
    float4 *p = reinterpret_cast<float4 *>(smem + elem);
    float4 v = *p;
    int mval = smask[row];
    int gc = n0 + col;
    v.x = (gc + 0 < mval) ? v.x : 0.f;
    v.y = (gc + 1 < mval) ? v.y : 0.f;
    v.z = (gc + 2 < mval) ? v.z : 0.f;
    v.w = (gc + 3 < mval) ? v.w : 0.f;
    *p = v;
  }
}

__device__ __forceinline__ void load_smask(int *smask, const int *mask, int m0) {
  if (threadIdx.x < TILE_M)
    smask[threadIdx.x] = mask[m0 + threadIdx.x];
}

// =============================================================================
// Reference
// =============================================================================
__global__ void mask_ref(const float *__restrict__ src, float *__restrict__ dst,
                         const int *__restrict__ mask, int M, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N)
    dst[(size_t)row * N + col] =
        (col < mask[row]) ? src[(size_t)row * N + col] : 0.f;
}

__global__ void copy_vec4(const float *__restrict__ src, float *__restrict__ dst,
                          int M, int N) {
  int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col4 * 4 < N)
    reinterpret_cast<float4 *>(dst + (size_t)row * N)[col4] =
        reinterpret_cast<const float4 *>(src + (size_t)row * N)[col4];
}

// =============================================================================
// Scheme 2 / LSU paths
// =============================================================================

// Fair coalesced baseline: 2D launch, one float4 per thread. Hardware TLP.
__global__ void mask_vec4_2d(const float *__restrict__ src,
                             float *__restrict__ dst, const int *__restrict__ mask,
                             int M, int N) {
  int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col4 * 4 >= N)
    return;
  int mval = mask[row];
  int gc = col4 * 4;
  float4 v = reinterpret_cast<const float4 *>(src + (size_t)row * N)[col4];
  float4 o;
  o.x = (gc + 0 < mval) ? v.x : 0.f;
  o.y = (gc + 1 < mval) ? v.y : 0.f;
  o.z = (gc + 2 < mval) ? v.z : 0.f;
  o.w = (gc + 3 < mval) ? v.w : 0.f;
  reinterpret_cast<float4 *>(dst + (size_t)row * N)[col4] = o;
}

// Streaming L2-only loads (do not pollute L1).
__global__ void mask_vec4_cg(const float *__restrict__ src,
                             float *__restrict__ dst, const int *__restrict__ mask,
                             int M, int N) {
  int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col4 * 4 >= N)
    return;
  int mval = mask[row];
  int gc = col4 * 4;
  const float *p = src + (size_t)row * N + gc;
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  float4 o;
  o.x = (gc + 0 < mval) ? v.x : 0.f;
  o.y = (gc + 1 < mval) ? v.y : 0.f;
  o.z = (gc + 2 < mval) ? v.z : 0.f;
  o.w = (gc + 3 < mval) ? v.w : 0.f;
  reinterpret_cast<float4 *>(dst + (size_t)row * N)[col4] = o;
}

// One block per row, threads walk N. Software pipelined float4 (scheme 2, fixed).
template <int DIST>
__global__ void mask_row_swp(const float *__restrict__ src,
                             float *__restrict__ dst, const int *__restrict__ mask,
                             int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;
  int mval = mask[row];
  const float4 *s = reinterpret_cast<const float4 *>(src + (size_t)row * N);
  float4 *d = reinterpret_cast<float4 *>(dst + (size_t)row * N);
  int n4 = N / 4;
  int tid = threadIdx.x;
  int niter = (n4 + blockDim.x - 1) / blockDim.x;

  float4 buf[DIST];
#pragma unroll
  for (int p = 0; p < DIST; ++p) {
    int idx = tid + p * blockDim.x;
    if (idx < n4)
      buf[p] = s[idx];
  }

  for (int it = 0; it < niter; ++it) {
    int idx = tid + it * blockDim.x;
    int slot = it % DIST;
    float4 v = buf[slot];
    if (idx < n4) {
      int gc = idx * 4;
      float4 o;
      o.x = (gc + 0 < mval) ? v.x : 0.f;
      o.y = (gc + 1 < mval) ? v.y : 0.f;
      o.z = (gc + 2 < mval) ? v.z : 0.f;
      o.w = (gc + 3 < mval) ? v.w : 0.f;
      d[idx] = o;
    }
    int nxt = idx + DIST * blockDim.x;
    if (nxt < n4)
      buf[slot] = s[nxt];
  }
}

// Same mapping, PTX prefetch.global.L2 instead of register rotation.
__global__ void mask_row_pref_ptx(const float *__restrict__ src,
                                  float *__restrict__ dst,
                                  const int *__restrict__ mask, int M, int N) {
  int row = blockIdx.x;
  if (row >= M)
    return;
  int mval = mask[row];
  const float4 *s = reinterpret_cast<const float4 *>(src + (size_t)row * N);
  float4 *d = reinterpret_cast<float4 *>(dst + (size_t)row * N);
  int n4 = N / 4;
  int tid = threadIdx.x;
  constexpr int DIST = 8;
  for (int idx = tid; idx < n4; idx += blockDim.x) {
    int pidx = idx + DIST * blockDim.x;
    if (pidx < n4)
      prefetch_l2(&s[pidx]);
    float4 v = s[idx];
    int gc = idx * 4;
    float4 o;
    o.x = (gc + 0 < mval) ? v.x : 0.f;
    o.y = (gc + 1 < mval) ? v.y : 0.f;
    o.z = (gc + 2 < mval) ? v.z : 0.f;
    o.w = (gc + 3 < mval) ? v.w : 0.f;
    d[idx] = o;
  }
}

// Original mapping: 1 thread owns 1 row, so a warp's loads are stride-N.
__global__ void mask_strided_row(const float *__restrict__ src,
                                 float *__restrict__ dst,
                                 const int *__restrict__ mask, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;
  int mval = mask[row];
  const float *s = src + (size_t)row * N;
  float *d = dst + (size_t)row * N;
  constexpr int DIST = 4;
  float4 buf[DIST];
  int n4 = N / 4;
#pragma unroll
  for (int p = 0; p < DIST; ++p) {
    if (p < n4)
      buf[p] = *reinterpret_cast<const float4 *>(s + p * 4);
  }
  for (int i = 0; i < n4; ++i) {
    int slot = i % DIST;
    float4 v = buf[slot];
    int gc = i * 4;
    float4 o;
    o.x = (gc + 0 < mval) ? v.x : 0.f;
    o.y = (gc + 1 < mval) ? v.y : 0.f;
    o.z = (gc + 2 < mval) ? v.z : 0.f;
    o.w = (gc + 3 < mval) ? v.w : 0.f;
    *reinterpret_cast<float4 *>(d + gc) = o;
    int nxt = i + DIST;
    if (nxt < n4)
      buf[slot] = *reinterpret_cast<const float4 *>(s + nxt * 4);
  }
}

// =============================================================================
// cp.async double-buffer (same 64x128 tile as TMA)
// =============================================================================
__global__ void mask_cpasync_s2(const float *__restrict__ src,
                                float *__restrict__ dst,
                                const int *__restrict__ mask, int M, int N) {
  extern __shared__ __align__(128) char raw[];
  float *smem[2] = {reinterpret_cast<float *>(raw),
                    reinterpret_cast<float *>(raw) + TILE_M * TILE_N};
  int *smask = reinterpret_cast<int *>(raw + 2 * TILE_BYTES);

  int m0 = blockIdx.y * TILE_M;
  int n_tiles = N / TILE_N;
  load_smask(smask, mask, m0);
  __syncthreads();

  auto issue = [&](int tile, int stage) {
    int n0 = tile * TILE_N;
    // 8192 floats / 256 threads = 32 floats = 8 x float4 per thread
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int e4 = threadIdx.x + i * THREADS;
      int elem = e4 * 4;
      int row = elem / TILE_N;
      int col = elem % TILE_N;
      const float *g = src + (size_t)(m0 + row) * N + n0 + col;
      cp_async_16B(smem[stage] + elem, g);
    }
    cp_async_commit();
  };

  issue(0, 0);
  int stage = 0;
  for (int t = 0; t < n_tiles; ++t) {
    int nxt = t + 1;
    if (nxt < n_tiles)
      issue(nxt, stage ^ 1);
    cp_async_wait_group(nxt < n_tiles ? 1 : 0);
    __syncthreads();
    apply_tile_stg(smem[stage], dst, smask, m0, t * TILE_N, N);
    __syncthreads();
    stage ^= 1;
  }
}

// =============================================================================
// TMA: one tile per block (TLP)
// =============================================================================
__global__ void __launch_bounds__(THREADS)
    mask_tma_one(const __grid_constant__ CUtensorMap tma_src, float *__restrict__ dst,
                 const int *__restrict__ mask, int M, int N) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  int *smask = reinterpret_cast<int *>(raw + TILE_BYTES + 16);

  int n0 = blockIdx.x * TILE_N;
  int m0 = blockIdx.y * TILE_M;

  if (threadIdx.x == 0)
    mbar_init(bar, 1);
  __syncthreads();
  load_smask(smask, mask, m0);

  if (threadIdx.x == 0) {
    mbar_expect_tx(bar, TILE_BYTES);
    tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma_src), n0, m0, bar);
  }
  mbar_wait(bar, 0);
  apply_tile_stg(smem, dst, smask, m0, n0, N);
}

__global__ void __launch_bounds__(THREADS)
    copy_tma_one(const __grid_constant__ CUtensorMap tma_src,
                 const __grid_constant__ CUtensorMap tma_dst) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);

  int n0 = blockIdx.x * TILE_N;
  int m0 = blockIdx.y * TILE_M;
  if (threadIdx.x == 0)
    mbar_init(bar, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect_tx(bar, TILE_BYTES);
    tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma_src), n0, m0, bar);
  }
  mbar_wait(bar, 0);
  tma_store_fence();
  if (threadIdx.x == 0) {
    tma_store_2d(smem, reinterpret_cast<uint64_t>(&tma_dst), n0, m0);
    tma_store_commit();
  }
  tma_store_wait();
}

__global__ void __launch_bounds__(THREADS)
    mask_tma_st(const __grid_constant__ CUtensorMap tma_src,
                const __grid_constant__ CUtensorMap tma_dst,
                const int *__restrict__ mask, int M, int N) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + TILE_BYTES);
  int *smask = reinterpret_cast<int *>(raw + TILE_BYTES + 16);

  int n0 = blockIdx.x * TILE_N;
  int m0 = blockIdx.y * TILE_M;
  if (threadIdx.x == 0)
    mbar_init(bar, 1);
  __syncthreads();
  load_smask(smask, mask, m0);
  if (threadIdx.x == 0) {
    mbar_expect_tx(bar, TILE_BYTES);
    tma_load_2d(smem, reinterpret_cast<uint64_t>(&tma_src), n0, m0, bar);
  }
  mbar_wait(bar, 0);
  apply_tile_inplace(smem, smask, n0);
  __syncthreads();
  tma_store_fence();
  if (threadIdx.x == 0) {
    tma_store_2d(smem, reinterpret_cast<uint64_t>(&tma_dst), n0, m0);
    tma_store_commit();
  }
  tma_store_wait();
}

// =============================================================================
// TMA pipeline along N: STAGES outstanding loads (the original proposal)
// grid.x = M / TILE_M, each block owns TILE_M rows x all N
// =============================================================================
template <int STAGES>
__global__ void __launch_bounds__(THREADS)
    mask_tma_pipe(const __grid_constant__ CUtensorMap tma_src,
                  float *__restrict__ dst, const int *__restrict__ mask, int M,
                  int N) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + STAGES * TILE_BYTES);
  int *smask =
      reinterpret_cast<int *>(raw + STAGES * TILE_BYTES + STAGES * 16);

  int m0 = blockIdx.x * TILE_M;
  int n_tiles = N / TILE_N;
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s)
      mbar_init(bar + s, 1);
  }
  __syncthreads();
  load_smask(smask, mask, m0);

  auto issue = [&](int tile, int stage) {
    if (threadIdx.x == 0) {
      mbar_expect_tx(bar + stage, TILE_BYTES);
      tma_load_2d(smem + stage * TILE_M * TILE_N,
                  reinterpret_cast<uint64_t>(&tma_src), tile * TILE_N, m0,
                  bar + stage);
    }
  };

  int pref = n_tiles < STAGES ? n_tiles : STAGES;
  for (int s = 0; s < pref; ++s)
    issue(s, s);
  __syncthreads();

  for (int t = 0; t < n_tiles; ++t) {
    int s = t % STAGES;
    uint32_t phase = (t / STAGES) & 1;
    mbar_wait(bar + s, phase);
    apply_tile_stg(smem + s * TILE_M * TILE_N, dst, smask, m0, t * TILE_N, N);
    __syncthreads();
    int nxt = t + STAGES;
    if (nxt < n_tiles)
      issue(nxt, s);
  }
}

// =============================================================================
// TMA concurrency microbench: serial vs fire-all-then-wait
// One block, K independent 64x128 tiles from well-separated rows.
// =============================================================================
template <int K>
__global__ void tma_outstanding(const __grid_constant__ CUtensorMap tma_src,
                                int mode, int row_stride, unsigned long long *clk) {
  extern __shared__ __align__(128) char raw[];
  float *smem = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  unsigned long long *t_smem =
      reinterpret_cast<unsigned long long *>(raw + K * TILE_BYTES + K * 16);

  if (threadIdx.x == 0) {
    for (int i = 0; i < K; ++i)
      mbar_init(bar + i, 1);
  }
  __syncthreads();

  unsigned long long t0, t1;
  if (threadIdx.x == 0)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
  __syncthreads();

  if (mode == 0) {
    for (int i = 0; i < K; ++i) {
      if (threadIdx.x == 0) {
        mbar_expect_tx(bar + i, TILE_BYTES);
        tma_load_2d(smem + i * TILE_M * TILE_N, reinterpret_cast<uint64_t>(&tma_src),
                    0, i * row_stride, bar + i);
      }
      mbar_wait(bar + i, 0);
    }
  } else {
    if (threadIdx.x == 0) {
      for (int i = 0; i < K; ++i) {
        mbar_expect_tx(bar + i, TILE_BYTES);
        tma_load_2d(smem + i * TILE_M * TILE_N, reinterpret_cast<uint64_t>(&tma_src),
                    0, i * row_stride, bar + i);
      }
    }
    for (int i = 0; i < K; ++i)
      mbar_wait(bar + i, 0);
  }

  // Observe smem so the wait is not dead. Sum is a compiler sink.
  float sink = smem[threadIdx.x];
#pragma unroll
  for (int i = 1; i < K; ++i)
    sink += smem[i * TILE_M * TILE_N + threadIdx.x];

  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : "f"(sink));
    *t_smem = t1 - t0;
    *clk = *t_smem;
  }
}

// =============================================================================
// Host
// =============================================================================
static CUtensorMap make_tma(void *p, int M, int N, int tm, int tn) {
  CUtensorMap d;
  cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)M};
  cuuint64_t str[1] = {(cuuint64_t)N * sizeof(float)};
  cuuint32_t box[2] = {(cuuint32_t)tn, (cuuint32_t)tm};
  cuuint32_t es[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, p, dims, str, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return d;
}

enum MaskPat { MASK_FULL, MASK_HALF, MASK_RAND, MASK_SPARSE };

static void fill_mask(int *h, int M, int N, MaskPat p, unsigned seed) {
  srand(seed);
  for (int i = 0; i < M; ++i) {
    switch (p) {
    case MASK_FULL:
      h[i] = N;
      break;
    case MASK_HALF:
      h[i] = N / 2;
      break;
    case MASK_SPARSE:
      h[i] = 16;
      break;
    case MASK_RAND:
      h[i] = rand() % (N + 1);
      break;
    }
  }
}

static const char *pat_name(MaskPat p) {
  switch (p) {
  case MASK_FULL:
    return "full";
  case MASK_HALF:
    return "half";
  case MASK_RAND:
    return "rand";
  case MASK_SPARSE:
    return "sparse16";
  }
  return "?";
}

struct Buf {
  float *src = nullptr, *dst = nullptr, *ref = nullptr;
  int *mask = nullptr;
  int M = 0, N = 0;
  void alloc(int m, int n) {
    free();
    M = m;
    N = n;
    size_t b = (size_t)m * n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&src, b));
    CUDA_CHECK(cudaMalloc(&dst, b));
    CUDA_CHECK(cudaMalloc(&ref, b));
    CUDA_CHECK(cudaMalloc(&mask, m * sizeof(int)));
  }
  void free() {
    if (src)
      cudaFree(src);
    if (dst)
      cudaFree(dst);
    if (ref)
      cudaFree(ref);
    if (mask)
      cudaFree(mask);
    src = dst = ref = nullptr;
    mask = nullptr;
  }
};

static void init_src(float *d, int M, int N) {
  std::vector<float> h((size_t)M * N);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c)
      h[(size_t)r * N + c] = r * 0.01f + c * 0.0001f;
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

template <typename Launch>
static float time_ms(Launch &&launch, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i)
    launch();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / reps;
}

struct Stats {
  float mean, med, stdev;
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
  float med = v[reps / 2];
  float var = 0.f;
  for (float x : v)
    var += (x - mean) * (x - mean);
  return {mean, med, sqrtf(var / reps)};
}

static bool check_vs_ref(float *dst, float *ref, int M, int N) {
  size_t b = (size_t)M * N * sizeof(float);
  std::vector<float> h(M * (size_t)N), r(M * (size_t)N);
  CUDA_CHECK(cudaMemcpy(h.data(), dst, b, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.data(), ref, b, cudaMemcpyDeviceToHost));
  int bad = 0;
  float mx = 0;
  size_t nn = (size_t)M * N;
  for (size_t i = 0; i < nn; ++i) {
    float d = fabsf(h[i] - r[i]);
    if (d > mx)
      mx = d;
    if (d > 1e-6f)
      ++bad;
  }
  if (bad) {
    printf("    FAIL  mismatches=%d / %zu  max_abs=%.3g\n", bad, nn, mx);
    return false;
  }
  return true;
}

static int smem_one() { return TILE_BYTES + 16 + TILE_M * (int)sizeof(int); }
static int smem_pipe(int stages) {
  return stages * TILE_BYTES + stages * 16 + TILE_M * (int)sizeof(int);
}
static int smem_cp() { return 2 * TILE_BYTES + TILE_M * (int)sizeof(int); }

static void set_dyn_smem(const void *fn, int bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(
      fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

static void print_occ(const char *name, const void *fn, int threads, int smem) {
  int nb = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fn, threads, smem));
  printf("  occupancy %-18s  %d blk/SM  smem=%d KB\n", name, nb, smem / 1024);
}

int main(int argc, char **argv) {
  CU_CHECK(cuInit(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::string which = argc > 1 ? argv[1] : std::string("all");
  auto want = [&](const char *tag) {
    return which == "all" || which == tag;
  };
  printf("================================================================\n");
  printf(" Mask TMA vs prefetch   GPU=%s  sm_%d%d  SMs=%d  L2=%d MB\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.l2CacheSize / 1024 / 1024);
  printf(" smem/SM=%zu KB  smem opt-in=%zu KB  bus=%d-bit\n",
         prop.sharedMemPerMultiprocessor / 1024, prop.sharedMemPerBlockOptin / 1024,
         prop.memoryBusWidth);
  printf(" tile=%dx%d  threads=%d  mode=%s\n", TILE_M, TILE_N, THREADS,
         which.c_str());
  printf("================================================================\n");

  set_dyn_smem((const void *)mask_tma_one, smem_one());
  set_dyn_smem((const void *)copy_tma_one, smem_one());
  set_dyn_smem((const void *)mask_tma_st, smem_one());
  set_dyn_smem((const void *)mask_tma_pipe<1>, smem_pipe(1));
  set_dyn_smem((const void *)mask_tma_pipe<2>, smem_pipe(2));
  set_dyn_smem((const void *)mask_tma_pipe<3>, smem_pipe(3));
  set_dyn_smem((const void *)mask_cpasync_s2, smem_cp());
  set_dyn_smem((const void *)tma_outstanding<1>, TILE_BYTES + 32);
  set_dyn_smem((const void *)tma_outstanding<2>, 2 * TILE_BYTES + 64);
  set_dyn_smem((const void *)tma_outstanding<3>, 3 * TILE_BYTES + 96);

  printf("\n--- occupancy ---\n");
  print_occ("vec4_2d", (const void *)mask_vec4_2d, 256, 0);
  print_occ("row_swp4", (const void *)mask_row_swp<4>, 128, 0);
  print_occ("strided", (const void *)mask_strided_row, 256, 0);
  print_occ("cpasync_s2", (const void *)mask_cpasync_s2, THREADS, smem_cp());
  print_occ("tma_one", (const void *)mask_tma_one, THREADS, smem_one());
  print_occ("tma_pipe1", (const void *)mask_tma_pipe<1>, THREADS, smem_pipe(1));
  print_occ("tma_pipe2", (const void *)mask_tma_pipe<2>, THREADS, smem_pipe(2));
  print_occ("tma_pipe3", (const void *)mask_tma_pipe<3>, THREADS, smem_pipe(3));
  print_occ("tma_st", (const void *)mask_tma_st, THREADS, smem_one());

  // ---------- 0. TMA outstanding requests ----------
  if (want("conc") || want("all")) {
  printf("\n=== Exp 0: TMA concurrent outstanding requests ===\n");
  printf("  one CTA, K tiles of %dx%d (%.1f KB each). serial=issue-wait-issue-wait\n",
         TILE_M, TILE_N, TILE_BYTES / 1024.0);
  printf("  concurrent=issue all, then wait all. row_stride=256 (separate L2 lines)\n");
  {
    const int MAXK = 4;
    const int rows = MAXK * 256 + TILE_M;
    const int cols = TILE_N;
    float *g = nullptr;
    CUDA_CHECK(cudaMalloc(&g, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(g, 1, (size_t)rows * cols * sizeof(float)));
    CUtensorMap desc = make_tma(g, rows, cols, TILE_M, TILE_N);
    unsigned long long *dclk = nullptr;
    CUDA_CHECK(cudaMalloc(&dclk, sizeof(unsigned long long)));

    auto runK = [&](auto kn, int K, const char *tag) {
      int dyn = K * TILE_BYTES + K * 16 + 16;
      auto measure = [&](int mode) {
        const int W = 20, R = 100;
        std::vector<unsigned long long> samp;
        samp.reserve(R);
        for (int i = 0; i < W; ++i)
          kn<<<1, THREADS, dyn>>>(desc, mode, 256, dclk);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < R; ++i) {
          kn<<<1, THREADS, dyn>>>(desc, mode, 256, dclk);
          CUDA_CHECK(cudaDeviceSynchronize());
          unsigned long long c;
          CUDA_CHECK(cudaMemcpy(&c, dclk, sizeof(c), cudaMemcpyDeviceToHost));
          samp.push_back(c);
        }
        std::sort(samp.begin(), samp.end());
        unsigned long long sum = 0;
        for (auto x : samp)
          sum += x;
        return std::make_tuple(samp.front(), samp[R / 2], samp.back(),
                               sum / samp.size());
      };
      auto [smin, smed, smax, smean] = measure(0);
      auto [cmin, cmed, cmax, cmean] = measure(1);
      printf("  K=%d %-6s  serial med=%7llu (min=%llu max=%llu)  "
             "conc med=%7llu (min=%llu max=%llu)  speedup=%.2fx\n",
             K, tag, smed, smin, smax, cmed, cmin, cmax,
             (double)smed / (double)cmed);
    };
    runK(tma_outstanding<1>, 1, "K=1");
    runK(tma_outstanding<2>, 2, "K=2");
    runK(tma_outstanding<3>, 3, "K=3");
    cudaFree(g);
    cudaFree(dclk);
  }
  } // exp 0

  auto bytes_rw = [](int M, int N) { return 2.0 * (double)M * N * sizeof(float); };

  dim3 vblock(32, 8);
  auto vgrid = [](int M, int N) {
    return dim3((N / 4 + 31) / 32, (M + 7) / 8);
  };

  // We'll use a lambda table via macros-ish launches inside the sweep.

  Buf buf;
  auto prep = [&](int M, int N, MaskPat pat) {
    buf.alloc(M, N);
    init_src(buf.src, M, N);
    std::vector<int> hm(M);
    fill_mask(hm.data(), M, N, pat, 42);
    CUDA_CHECK(cudaMemcpy(buf.mask, hm.data(), M * sizeof(int),
                          cudaMemcpyHostToDevice));
    dim3 b(32, 8);
    dim3 g((N + 31) / 32, (M + 7) / 8);
    mask_ref<<<g, b>>>(buf.src, buf.ref, buf.mask, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(buf.dst, 0xAB, (size_t)M * N * sizeof(float)));
  };

  struct Row {
    const char *name;
    float ms;
    double gbs;
  };

  auto report = [&](const char *name, float ms, int M, int N) {
    double gbs = bytes_rw(M, N) / 1e9 / (ms / 1e3);
    printf("  %-16s  %8.3f ms  %7.1f GB/s\n", name, ms, gbs);
    return Row{name, ms, gbs};
  };

  auto launch_all = [&](int M, int N, bool do_check, int warmup, int reps,
                        std::vector<Row> *out) {
    CUtensorMap tsrc = make_tma(buf.src, M, N, TILE_M, TILE_N);
    CUtensorMap tdst = make_tma(buf.dst, M, N, TILE_M, TILE_N);
    dim3 vg = vgrid(M, N);
    dim3 tg(N / TILE_N, M / TILE_M);
    dim3 tb(THREADS);
    int s_one = smem_one(), s2 = smem_pipe(2), s1 = smem_pipe(1), s3 = smem_pipe(3),
        scp = smem_cp();

    auto run = [&](const char *name, auto &&kn) {
      bool is_copy = strncmp(name, "copy_", 5) == 0;
      if (do_check && !is_copy)
        CUDA_CHECK(cudaMemset(buf.dst, 0xAB, (size_t)M * N * sizeof(float)));
      float ms = time_ms(kn, warmup, reps);
      if (do_check && !is_copy) {
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!check_vs_ref(buf.dst, buf.ref, M, N))
          printf("    ^^ %s\n", name);
      }
      Row r = report(name, ms, M, N);
      if (out)
        out->push_back(r);
    };

    run("copy_vec4", [&] {
      copy_vec4<<<vg, vblock>>>(buf.src, buf.dst, M, N);
    });
    run("copy_tma", [&] {
      copy_tma_one<<<tg, tb, s_one>>>(tsrc, tdst);
    });
    run("vec4_2d", [&] {
      mask_vec4_2d<<<vg, vblock>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("vec4_cg", [&] {
      mask_vec4_cg<<<vg, vblock>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("row_swp1", [&] {
      mask_row_swp<1><<<M, 128>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("row_swp4", [&] {
      mask_row_swp<4><<<M, 128>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("row_swp8", [&] {
      mask_row_swp<8><<<M, 128>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("row_pref_ptx", [&] {
      mask_row_pref_ptx<<<M, 128>>>(buf.src, buf.dst, buf.mask, M, N);
    });
    run("strided_row", [&] {
      int blk = 256;
      mask_strided_row<<<(M + blk - 1) / blk, blk>>>(buf.src, buf.dst, buf.mask,
                                                     M, N);
    });
    run("cpasync_s2", [&] {
      mask_cpasync_s2<<<dim3(1, M / TILE_M), tb, scp>>>(buf.src, buf.dst,
                                                        buf.mask, M, N);
    });
    run("tma_one", [&] {
      mask_tma_one<<<tg, tb, s_one>>>(tsrc, buf.dst, buf.mask, M, N);
    });
    run("tma_st", [&] {
      mask_tma_st<<<tg, tb, s_one>>>(tsrc, tdst, buf.mask, M, N);
    });
    run("tma_pipe1", [&] {
      mask_tma_pipe<1><<<M / TILE_M, tb, s1>>>(tsrc, buf.dst, buf.mask, M, N);
    });
    run("tma_pipe2", [&] {
      mask_tma_pipe<2><<<M / TILE_M, tb, s2>>>(tsrc, buf.dst, buf.mask, M, N);
    });
    run("tma_pipe3", [&] {
      mask_tma_pipe<3><<<M / TILE_M, tb, s3>>>(tsrc, buf.dst, buf.mask, M, N);
    });
  };

  struct Sz {
    int M, N;
  };

  // ---------- 1. correctness ----------
  if (want("check") || want("all")) {
  printf("\n=== Exp 1: correctness (256 x 512, mask=rand) ===\n");
  prep(256, 512, MASK_RAND);
  launch_all(256, 512, true, 1, 1, nullptr);
  printf("  (FAIL lines only if a mask kernel mismatches ref)\n");
  }

  // ---------- 2. size sweep, full mask (pure copy + compare) ----------
  if (want("size") || want("all")) {
  printf("\n=== Exp 2: size sweep  mask=full  (read+write GB/s) ===\n");
  printf("  note: working set = 2*M*N*4; L2=%d MB. Below that, numbers are L2 not HBM.\n",
         prop.l2CacheSize / 1024 / 1024);
  Sz sizes[] = {{64, 128},   {64, 1024},  {128, 256}, {256, 256},
                {256, 1024}, {512, 512},  {512, 4096}, {1024, 1024},
                {2048, 2048}, {4096, 512}, {4096, 4096}, {8192, 8192}};
  for (auto sz : sizes) {
    int M = sz.M, N = sz.N;
    int reps = (M * (long)N >= 4096L * 4096) ? 20 : (M * (long)N >= 1024L * 1024) ? 50 : 200;
    int warm = (reps >= 50) ? 10 : 20;
    double ws_mb = 2.0 * M * N * 4 / 1e6;
    const char *bound = ws_mb * 1e6 < prop.l2CacheSize ? "L2-ish" : "HBM";
    printf("\n-- %d x %d  (%.2f MB src, WS=%.1f MB %s)  warm=%d reps=%d --\n",
           M, N, M * (double)N * 4 / 1e6, ws_mb, bound, warm, reps);
    prep(M, N, MASK_FULL);
    launch_all(M, N, false, warm, reps, nullptr);
  }
  }

  // ---------- 3. mask pattern @ 4096^2 ----------
  if (want("pat") || want("all")) {
  printf("\n=== Exp 3: mask pattern @ 4096x4096 ===\n");
  for (MaskPat p : {MASK_FULL, MASK_HALF, MASK_RAND, MASK_SPARSE}) {
    printf("\n-- pattern=%s --\n", pat_name(p));
    prep(4096, 4096, p);
    launch_all(4096, 4096, false, 10, 20, nullptr);
  }
  }

  // ---------- 4. tiny-tile / launch overhead ----------
  if (want("tiny") || want("all")) {
  printf("\n=== Exp 4: tiny working set (TMA setup vs LSU) ===\n");
  for (auto sz : {Sz{64, 128}, Sz{64, 256}, Sz{128, 128}, Sz{256, 128}}) {
    printf("\n-- %d x %d --\n", sz.M, sz.N);
    prep(sz.M, sz.N, MASK_FULL);
    launch_all(sz.M, sz.N, false, 30, 300, nullptr);
  }
  }

  // ---------- 5. stable HBM median (working set >> L2) ----------
  if (want("hbm") || want("all")) {
    printf("\n=== Exp 5: stable HBM  (median of per-launch events) ===\n");
    struct HK {
      const char *name;
      int M, N;
    };
    for (HK hk : {HK{"4096^2", 4096, 4096}, HK{"8192^2", 8192, 8192}}) {
      prep(hk.M, hk.N, MASK_FULL);
      CUtensorMap tsrc = make_tma(buf.src, hk.M, hk.N, TILE_M, TILE_N);
      CUtensorMap tdst = make_tma(buf.dst, hk.M, hk.N, TILE_M, TILE_N);
      dim3 vg = vgrid(hk.M, hk.N);
      dim3 tg(hk.N / TILE_N, hk.M / TILE_M);
      dim3 tb(THREADS);
      int s_one = smem_one(), s2 = smem_pipe(2), s1 = smem_pipe(1), scp = smem_cp();
      int warm = 10, reps = 30;
      printf("\n-- %s  WS=%.1f MB --\n", hk.name,
             2.0 * hk.M * hk.N * 4 / 1e6);
      auto show = [&](const char *name, auto &&kn) {
        Stats s = time_stats(kn, warm, reps);
        double gbs = bytes_rw(hk.M, hk.N) / 1e9 / (s.med / 1e3);
        printf("  %-16s  med=%7.3f ms  mean=%7.3f  std=%5.3f  %6.1f GB/s\n",
               name, s.med, s.mean, s.stdev, gbs);
      };
      show("copy_vec4", [&] { copy_vec4<<<vg, vblock>>>(buf.src, buf.dst, hk.M, hk.N); });
      show("copy_tma", [&] { copy_tma_one<<<tg, tb, s_one>>>(tsrc, tdst); });
      show("vec4_2d", [&] {
        mask_vec4_2d<<<vg, vblock>>>(buf.src, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("vec4_cg", [&] {
        mask_vec4_cg<<<vg, vblock>>>(buf.src, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("row_swp1", [&] {
        mask_row_swp<1><<<hk.M, 128>>>(buf.src, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("row_swp4", [&] {
        mask_row_swp<4><<<hk.M, 128>>>(buf.src, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("row_pref_ptx", [&] {
        mask_row_pref_ptx<<<hk.M, 128>>>(buf.src, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("strided_row", [&] {
        mask_strided_row<<<(hk.M + 255) / 256, 256>>>(buf.src, buf.dst, buf.mask,
                                                      hk.M, hk.N);
      });
      show("cpasync_s2", [&] {
        mask_cpasync_s2<<<dim3(1, hk.M / TILE_M), tb, scp>>>(buf.src, buf.dst,
                                                             buf.mask, hk.M, hk.N);
      });
      show("tma_one", [&] {
        mask_tma_one<<<tg, tb, s_one>>>(tsrc, buf.dst, buf.mask, hk.M, hk.N);
      });
      show("tma_st", [&] {
        mask_tma_st<<<tg, tb, s_one>>>(tsrc, tdst, buf.mask, hk.M, hk.N);
      });
      show("tma_pipe1", [&] {
        mask_tma_pipe<1><<<hk.M / TILE_M, tb, s1>>>(tsrc, buf.dst, buf.mask, hk.M,
                                                    hk.N);
      });
      show("tma_pipe2", [&] {
        mask_tma_pipe<2><<<hk.M / TILE_M, tb, s2>>>(tsrc, buf.dst, buf.mask, hk.M,
                                                    hk.N);
      });
    }
  }

  if (want("prof")) {
    printf("\n=== ncu-friendly 4096x4096  (3 kernels, 5 launches) ===\n");
    prep(4096, 4096, MASK_FULL);
    CUtensorMap tsrc = make_tma(buf.src, 4096, 4096, TILE_M, TILE_N);
    dim3 vg = vgrid(4096, 4096);
    dim3 tg(4096 / TILE_N, 4096 / TILE_M);
    for (int i = 0; i < 5; ++i)
      mask_vec4_2d<<<vg, vblock>>>(buf.src, buf.dst, buf.mask, 4096, 4096);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 5; ++i)
      mask_tma_one<<<tg, dim3(THREADS), smem_one()>>>(tsrc, buf.dst, buf.mask, 4096,
                                                      4096);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 5; ++i)
      mask_strided_row<<<(4096 + 255) / 256, 256>>>(buf.src, buf.dst, buf.mask, 4096,
                                                    4096);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  launched vec4_2d / tma_one / strided_row x5\n");
  }

  buf.free();
  printf("\n================================================================\n");
  printf("done.\n");
  return 0;
}
