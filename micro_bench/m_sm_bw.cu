// =============================================================================
// m_sm_bw — 不满 SM / persistent CTA 能否打满 DRAM 带宽
//
// 编译:
//   nvcc -O3 -std=c++17 -arch=sm_120 -lcuda -o /tmp/m_sm_bw micro_bench/m_sm_bw.cu
//
// 2026-09-03  RTX 5050 Laptop / sm_120 / 20 SM / 32 MB L2 / 128-bit GDDR7
//   theo 384 GB/s, LSU roof 353 GB/s (92%). 这是 GDDR7, 不是 H20 的 ~3 TB/s HBM.
//
//   1 CTA/SM LSU  : ~12 GB/s/SM, 20 SM 才 235 GB/s (67% roof) — 打不满
//   1 CTA/SM TMA  : ~60 GB/s/SM, **8 SM 就 359 GB/s (101%)** — 打满
//   1 SM 打满 occupancy (6 CTA / 32 warps) LSU: 72 GB/s (20% roof) — 单 SM 打不满
//   6~8 SM × 满 occupancy LSU: 94~98% roof
//   grid=50 (2.5 CTA/SM, 20 SM 全开) LSU 358 / TMA 371 GB/s — 打满
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
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

static constexpr int TILE_M = 32;
static constexpr int TILE_N = 128;
static constexpr int TILE_ELEMS = TILE_M * TILE_N;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(float);
static constexpr int TMA_STAGES = 6;

// -----------------------------------------------------------------------------
// TMA / mbarrier helpers (same PTX that already works on this 5050)
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvta_to_shared(const void *p) {
  uint32_t a;
  asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
               : "=r"(a)
               : "l"(p));
  return a;
}
__device__ __forceinline__ unsigned smid() {
  unsigned x;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
  return x;
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
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_"
      "tx::bytes [%0], [%1, {%3, %4}], [%2];\n" ::"r"(cvta_to_shared(dst)),
      "l"(desc), "r"(cvta_to_shared(bar)), "r"(cx), "r"(cy)
      : "memory");
}

// -----------------------------------------------------------------------------
// LSU streaming read. 1 CTA/SM when launched with launch_bounds(256,1).
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1)
    lsu_rd_p1(const float4 *__restrict__ g, size_t n4, int iters, float *sink) {
  float4 acc = make_float4(0, 0, 0, 0);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int t = 0; t < iters; ++t) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      float4 v = g[i];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x + acc.y + acc.z + acc.w == 1e30f)
    *sink = acc.x;
}

// High occupancy: 256 thr, 6 CTA/SM = 1536 threads (typical max on this SKU).
__global__ void __launch_bounds__(256, 6)
    lsu_rd_occ(const float4 *__restrict__ g, size_t n4, int iters, float *sink) {
  float4 acc = make_float4(0, 0, 0, 0);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int t = 0; t < iters; ++t) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
      float4 v = g[i];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x + acc.y + acc.z + acc.w == 1e30f)
    *sink = acc.x;
}

// One CTA, variable warps: answers "单 SM 单 CTA 能推多少".
__global__ void lsu_rd_1cta(const float4 *__restrict__ g, size_t n4, int iters,
                            float *sink) {
  float4 acc = make_float4(0, 0, 0, 0);
  for (int t = 0; t < iters; ++t) {
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x) {
      float4 v = g[i];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x + acc.y + acc.z + acc.w == 1e30f)
    *sink = acc.x;
}

// Pack many CTAs onto N SMs: launch SM*cta_per_sm, idle the rest via %smid.
// Each working CTA streams a private slice so they do not L2-hit each other.
__global__ void __launch_bounds__(256, 6)
    lsu_rd_nsm(const float4 *__restrict__ g, size_t n4_slice, int n_sms,
               int iters, float *sink) {
  if (smid() >= (unsigned)n_sms)
    return;
  const float4 *my = g + (size_t)blockIdx.x * n4_slice;
  float4 acc = make_float4(0, 0, 0, 0);
  for (int t = 0; t < iters; ++t) {
    for (size_t i = threadIdx.x; i < n4_slice; i += blockDim.x) {
      float4 v = my[i];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
  }
  if (acc.x + acc.y + acc.z + acc.w == 1e30f)
    *sink = acc.x;
}

__global__ void k_smid(int *hist, int *nblk) {
  if (threadIdx.x == 0) {
    atomicAdd(hist + smid(), 1);
    atomicAdd(nblk, 1);
  }
}

// -----------------------------------------------------------------------------
// TMA persistent: each CTA pipelines STAGES outstanding 32KB bulk loads.
// launch_bounds 1 → grid<=SM 时每 SM 一个 TMA 流水线。
// -----------------------------------------------------------------------------
template <int STAGES, int MAX_CTA>
__global__ void __launch_bounds__(256, MAX_CTA)
    tma_rd(const __grid_constant__ CUtensorMap tma, int n_tiles, int iters,
           float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + STAGES * TILE_BYTES);

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s)
      mbar_init(bar + s, 1);
  }
  __syncthreads();

  float acc = 0.f;
  int group = 0;
  for (int it = 0; it < iters; ++it) {
    int n_mine = 0;
    for (int t = blockIdx.x; t < n_tiles; t += gridDim.x)
      ++n_mine;
    n_mine -= n_mine % STAGES;
    for (int base = 0; base < n_mine; base += STAGES, ++group) {
      int batch = n_mine - base;
      if (batch > STAGES)
        batch = STAGES;
      uint32_t phase = group & 1;
      if (threadIdx.x == 0) {
        for (int i = 0; i < batch; ++i) {
          int tile = blockIdx.x + (base + i) * (int)gridDim.x;
          mbar_expect_tx(bar + i, TILE_BYTES);
          tma_load_2d(sm + i * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma), 0,
                      tile * TILE_M, bar + i);
        }
      }
      __syncthreads();
      for (int i = 0; i < batch; ++i)
        mbar_wait(bar + i, phase);
      acc += sm[threadIdx.x];
    }
  }
  if (acc == 1e30f)
    *sink = acc;
}

// TMA onto N SMs only. Each working CTA owns a contiguous tile range.
template <int STAGES>
__global__ void __launch_bounds__(256, 4)
    tma_rd_nsm(const __grid_constant__ CUtensorMap tma, int tiles_per_cta,
               int n_sms, int iters, float *sink) {
  if (smid() >= (unsigned)n_sms)
    return;
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + STAGES * TILE_BYTES);
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s)
      mbar_init(bar + s, 1);
  }
  __syncthreads();

  const int n_mine = tiles_per_cta - (tiles_per_cta % STAGES);
  const int tile0 = blockIdx.x * tiles_per_cta;
  float acc = 0.f;
  int group = 0;
  for (int it = 0; it < iters; ++it) {
    for (int base = 0; base < n_mine; base += STAGES, ++group) {
      int batch = tiles_per_cta - base;
      if (batch > STAGES)
        batch = STAGES;
      uint32_t phase = group & 1;
      if (threadIdx.x == 0) {
        for (int i = 0; i < batch; ++i) {
          mbar_expect_tx(bar + i, TILE_BYTES);
          tma_load_2d(sm + i * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma), 0,
                      (tile0 + base + i) * TILE_M, bar + i);
        }
      }
      __syncthreads();
      for (int i = 0; i < batch; ++i)
        mbar_wait(bar + i, phase);
      acc += sm[threadIdx.x];
    }
  }
  if (acc == 1e30f)
    *sink = acc;
}

// -----------------------------------------------------------------------------
// Host
// -----------------------------------------------------------------------------
static CUtensorMap make_tma(void *p, int M, int N) {
  CUtensorMap d;
  cuuint64_t dims[2] = {(cuuint64_t)N, (cuuint64_t)M};
  cuuint64_t str[1] = {(cuuint64_t)N * sizeof(float)};
  cuuint32_t box[2] = {(cuuint32_t)TILE_N, (cuuint32_t)TILE_M};
  cuuint32_t es[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, p, dims, str, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return d;
}

template <typename Launch>
static float median_ms(int warmup, int reps, Launch &&launch) {
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
  return v[reps / 2];
}

static int unique_sms(int grid, int thr, int *hist_out, int max_sm) {
  int *dhist = nullptr, *dn = nullptr;
  CUDA_CHECK(cudaMalloc(&dhist, max_sm * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dn, 4));
  CUDA_CHECK(cudaMemset(dhist, 0, max_sm * sizeof(int)));
  CUDA_CHECK(cudaMemset(dn, 0, 4));
  k_smid<<<grid, thr>>>(dhist, dn);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int> h(max_sm);
  CUDA_CHECK(cudaMemcpy(h.data(), dhist, max_sm * sizeof(int),
                        cudaMemcpyDeviceToHost));
  cudaFree(dhist);
  cudaFree(dn);
  int u = 0;
  for (int i = 0; i < max_sm; ++i) {
    if (hist_out)
      hist_out[i] = h[i];
    if (h[i])
      ++u;
  }
  return u;
}

int main() {
  CU_CHECK(cuInit(0));
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int SM = prop.multiProcessorCount;
  const double l2_mb = prop.l2CacheSize / 1048576.0;
  int mem_clk_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0));
  const double theo = (double)mem_clk_khz * 1e3 * 2.0 *
                      (prop.memoryBusWidth / 8.0) / 1e9;

  printf("================================================================\n");
  printf(" m_sm_bw  %s  sm_%d%d  SM=%d  L2=%.0f MB  bus=%d-bit\n", prop.name,
         prop.major, prop.minor, SM, l2_mb, prop.memoryBusWidth);
  printf(" memClock=%.3f GHz  theo DRAM (clk*2*bus/8) = %.1f GB/s\n",
         mem_clk_khz / 1e6, theo);
  printf(" NOTE: this 5050 is GDDR7, not HBM. Peak is ~%.0f GB/s class,\n",
         theo);
  printf("       NOT the ~3 TB/s HBM number from H20. Question is the same:\n");
  printf("       can a subset of SMs saturate the *chip* DRAM roof?\n");
  printf("================================================================\n");

  int occ_p1 = 0, occ_occ = 0, occ_tma = 0, occ_1cta1024 = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_p1, (const void *)lsu_rd_p1, 256, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_occ, (const void *)lsu_rd_occ, 256, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ_1cta1024, (const void *)lsu_rd_1cta, 1024, 0));

  int maxsmem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &maxsmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  const int tma_smem = TMA_STAGES * TILE_BYTES + TMA_STAGES * 16;
  if (tma_smem > maxsmem) {
    fprintf(stderr, "TMA smem %d > max %d\n", tma_smem, maxsmem);
    return 1;
  }
  printf("\nmax dynamic smem/block = %d KB; TMA smem = %d KB (stages=%d, "
         "tile=%d KB)\n",
         maxsmem / 1024, tma_smem / 1024, TMA_STAGES, TILE_BYTES / 1024);

  auto *tma_p1 = (const void *)tma_rd<TMA_STAGES, 1>;
  auto *tma_occ = (const void *)tma_rd<TMA_STAGES, 4>;
  auto *tma_nsm = (const void *)tma_rd_nsm<TMA_STAGES>;
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_p1, cudaFuncAttributeMaxDynamicSharedMemorySize, tma_smem));
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_occ, cudaFuncAttributeMaxDynamicSharedMemorySize, tma_smem));
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_nsm, cudaFuncAttributeMaxDynamicSharedMemorySize, tma_smem));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_tma, tma_p1, 256,
                                                           tma_smem));
  int occ_tma8 = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_tma8, tma_occ,
                                                           256, tma_smem));
  printf("occupancy: lsu_p1=%d  lsu_occ=%d  lsu_1cta1024=%d  tma_p1=%d  "
         "tma_occ=%d  CTA/SM\n",
         occ_p1, occ_occ, occ_1cta1024, occ_tma, occ_tma8);

  // 512 MB >> 32 MB L2. 2 iters already ~1 ms at peak; scale for 1-SM runs.
  const size_t BYTES = 512ull << 20;
  const size_t n4 = BYTES / 16;
  const int Ncols = TILE_N;
  const int Md = (int)(BYTES / (Ncols * sizeof(float))); // 1,048,576
  const int n_tiles = Md / TILE_M;                       // 16384
  float *A = nullptr, *B = nullptr, *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&A, BYTES));
  CUDA_CHECK(cudaMalloc(&B, BYTES));
  CUDA_CHECK(cudaMalloc(&sink, 4));
  CUDA_CHECK(cudaMemset(A, 1, BYTES));
  CUtensorMap tA = make_tma(A, Md, Ncols);

  printf("TMA smoke (1 CTA, 1 batch)...\n");
  tma_rd<TMA_STAGES, 1><<<1, 256, tma_smem>>>(tA, TMA_STAGES, 1, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("TMA smoke ok\n");

  auto gbs = [&](double bytes, float ms) {
    return bytes / 1e9 / (ms / 1e3);
  };

  // Boost clocks: a few fat warmups before any timed sweep.
  for (int i = 0; i < 8; ++i)
    lsu_rd_occ<<<SM * 6, 256>>>((float4 *)A, n4, 2, sink);
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("\n=== 0. chip roof (all SMs, high occ) + copy engine ===\n");
  float ms_peak = median_ms(3, 7, [&] {
    lsu_rd_occ<<<SM * 6, 256>>>((float4 *)A, n4, 4, sink);
  });
  double peak = gbs((double)BYTES * 4, ms_peak);
  printf("  LSU full  grid=%d*6  4iters  %7.2f ms  %7.1f GB/s  (%.0f%% theo)\n",
         SM, ms_peak, peak, 100.0 * peak / theo);

  float ms_ce = median_ms(3, 7, [&] {
    CUDA_CHECK(cudaMemcpy(B, A, BYTES, cudaMemcpyDeviceToDevice));
  });
  // D2D is read+write on the same bus; count both directions (matches lsu_tma_bw).
  printf("  cudaMemcpy D2D (R+W)     %7.2f ms  %7.1f GB/s  (%.0f%% theo)  "
         "[copy engine, no SM]\n",
         ms_ce, gbs(2.0 * BYTES, ms_ce),
         100.0 * gbs(2.0 * BYTES, ms_ce) / theo);

  const double roof = peak; // use measured LSU peak as 100%

  printf("\n=== smid map (does grid=N really sit on N SMs?) ===\n");
  {
    std::vector<int> grids = {1, 2, 4, SM / 2, SM, 50, SM * 2, SM * 6};
    printf("  %6s %8s %10s  blk/SM histogram (nonzero)\n", "grid", "uniqSM",
           "max/SM");
    std::vector<int> hist(SM);
    for (int g : grids) {
      if (g <= 0)
        continue;
      std::fill(hist.begin(), hist.end(), 0);
      int u = unique_sms(g, 256, hist.data(), SM);
      int mx = 0;
      for (int x : hist)
        mx = std::max(mx, x);
      printf("  %6d %8d %10d  ", g, u, mx);
      int shown = 0;
      for (int i = 0; i < SM && shown < 8; ++i)
        if (hist[i]) {
          printf("SM%d:%d ", i, hist[i]);
          ++shown;
        }
      printf("%s\n", u > 8 ? "..." : "");
    }
  }

  // Scale iters so 1-SM runs are not launch-noise, full-SM runs not minutes.
  auto iters_for = [&](int nthreads) {
    double frac = std::min(1.0, nthreads / (double)(SM * 1536));
    double expect = std::max(25.0, roof * std::max(frac, 0.05));
    int it = (int)(0.03 * expect * 1e9 / (double)BYTES);
    return std::max(2, std::min(it, 48));
  };

  printf("\n=== A. persistent 1 CTA/SM  (grid<=SM → 每 SM 1 block; grid=50 → ~2.5/SM) ===\n");
  printf("  lsu_rd_p1 occ=%d CTA/SM, tma_p1 occ=%d (TMA smem pins 1 CTA/SM)\n",
         occ_p1, occ_tma);
  printf("  %6s %7s %7s %8s %8s %8s %8s %7s %7s\n", "grid", "uniqSM", "iters",
         "LSU", "TMA", "LSU/SM", "TMA/SM", "%roof", "TMA%");
  printf("  %6s %7s %7s %8s %8s %8s %8s %7s %7s\n", "", "", "", "GB/s", "GB/s",
         "GB/s", "GB/s", "LSU", "roof");

  std::vector<int> sweep;
  for (int g = 1; g <= SM; ++g) {
    if (g <= 8 || g == SM || g == SM / 2 || g % 2 == 0)
      sweep.push_back(g);
  }
  sweep.push_back(50);
  sweep.push_back(SM * 2);
  std::sort(sweep.begin(), sweep.end());
  sweep.erase(std::unique(sweep.begin(), sweep.end()), sweep.end());

  std::vector<int> hist(SM);
  for (int g : sweep) {
    int u = unique_sms(g, 256, hist.data(), SM);
    int it = iters_for(g * 256);
    float ms_l = median_ms(2, 5, [&] {
      lsu_rd_p1<<<g, 256>>>((float4 *)A, n4, it, sink);
    });
    float ms_t = median_ms(2, 5, [&] {
      tma_rd<TMA_STAGES, 1><<<g, 256, tma_smem>>>(tA, n_tiles, it, sink);
    });
    double bl = (double)BYTES * it;
    double l = gbs(bl, ms_l), t = gbs(bl, ms_t);
    printf("  %6d %7d %7d %8.1f %8.1f %8.1f %8.1f %6.0f%% %6.0f%%\n", g, u, it,
           l, t, l / u, t / u, 100.0 * l / roof, 100.0 * t / roof);
  }

  printf("\n=== B. persistent 50 / SM / SM*k  CTAs, occupancy NOT capped to 1 ===\n");
  printf("  %8s %7s %7s %8s %8s %7s\n", "grid", "uniqSM", "cta/SM", "LSU GB/s",
         "TMA GB/s", "%roof");
  std::vector<int> grids_b = {1, 4, SM / 2, SM, 50, SM * 2, SM * 4, SM * 6};
  for (int g : grids_b) {
    if (g <= 0)
      continue;
    int u = unique_sms(g, 256, hist.data(), SM);
    int it = iters_for(g * 256);
    float ms_l = median_ms(2, 5, [&] {
      lsu_rd_occ<<<g, 256>>>((float4 *)A, n4, it, sink);
    });
    float ms_t = median_ms(2, 5, [&] {
      tma_rd<TMA_STAGES, 4><<<g, 256, tma_smem>>>(tA, n_tiles, it, sink);
    });
    double bl = (double)BYTES * it;
    printf("  %8d %7d %7.2f %8.1f %8.1f %6.0f%%\n", g, u, (double)g / u,
           gbs(bl, ms_l), gbs(bl, ms_t), 100.0 * gbs(bl, ms_l) / roof);
  }

  printf("\n=== C. 1 CTA on 1 SM, scale warps (grid=1) ===\n");
  printf("  %8s %8s %8s\n", "threads", "warps", "GB/s");
  for (int thr : {32, 64, 128, 256, 512, 768, 1024}) {
    int it = iters_for(thr);
    float ms = median_ms(2, 5, [&] {
      lsu_rd_1cta<<<1, thr>>>((float4 *)A, n4, it, sink);
    });
    printf("  %8d %8d %8.1f\n", thr, thr / 32, gbs((double)BYTES * it, ms));
  }

  printf("\n=== D. pin N SMs, each with high occupancy (smid filter) ===\n");
  printf("  launch grid=SM*cta_ps, only smid<N work. private 8MB slices.\n");
  printf("  TMA 列在 N 较小时 working set 贴近 L2, 会虚高; TMA 缩放看 section A.\n");
  const size_t slice_bytes = 8ull << 20;
  const size_t n4_slice = slice_bytes / 16;
  const int cta_ps = 6;
  const int tiles_per_cta = (int)(slice_bytes / TILE_BYTES); // 256
  // buffer must cover SM*cta_ps slices
  size_t need = (size_t)SM * cta_ps * slice_bytes;
  float *C = nullptr;
  CUDA_CHECK(cudaMalloc(&C, need));
  CUDA_CHECK(cudaMemset(C, 1, need));
  const int Md2 = (int)(need / (Ncols * sizeof(float)));
  CUtensorMap tC = make_tma(C, Md2, Ncols);

  printf("  %6s %8s %8s %8s %8s %7s\n", "N_SM", "CTAs", "LSU", "TMA", "LSU/SM",
         "%roof");
  std::vector<int> nsms = {1, 2, 4, 8, 12, 16, SM};
  if (SM >= 20)
    nsms = {1, 2, 4, 6, 8, 10, 12, 16, 20, SM};
  for (int n : nsms) {
    if (n > SM)
      continue;
    int grid = SM * cta_ps;
    int work_cta = n * cta_ps; // expected if 1-spread then filter
    int it = 8;
    float ms_l = median_ms(2, 5, [&] {
      lsu_rd_nsm<<<grid, 256>>>((float4 *)C, n4_slice, n, it, sink);
    });
    float ms_t = median_ms(2, 5, [&] {
      tma_rd_nsm<TMA_STAGES>
          <<<grid, 256, tma_smem>>>(tC, tiles_per_cta, n, it, sink);
    });
    // Only smid<n CTAs do work. Scheduler typically puts cta_ps on every SM,
    // so working CTAs ≈ n * cta_ps.
    double bytes_l = (double)work_cta * slice_bytes * it;
    double l = gbs(bytes_l, ms_l), t = gbs(bytes_l, ms_t);
    printf("  %6d %8d %8.1f %8.1f %8.1f %6.0f%%\n", n, work_cta, l, t, l / n,
           100.0 * l / roof);
  }

  printf("\n================================================================\n");
  printf(" roof = %.1f GB/s (LSU all-SM). theo = %.1f GB/s.\n", roof, theo);
  printf(" Little's law: BW ≈ outstanding_bytes / DRAM_latency.\n");
  printf(" 1 SM 的 LSU/TMA miss queue 有限; 卡级 DRAM 要足够多 outstanding\n");
  printf(" 才能打满. 上表 %%roof vs N_SM 就是这条曲线。\n");
  printf("================================================================\n");
  return 0;
}
