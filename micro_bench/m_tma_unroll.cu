// =============================================================================
// m_tma_unroll — 1 个 SM 里并行 outstanding 多发 TMA，能不能打满 DRAM
//
// 2026-09-03 RTX 5050 / 20 SM / GDDR 屋顶 ~353 GB/s
//   1 SM 扫 K:  1→~10, 2→~22, 4→~43, 8→~65, 12→~64 GB/s
//   最佳: 8 个 warp 各发一条 TMA (kwarp K=8) ≈ 65 GB/s = 18% 屋顶
//   K 从 8 再加到 12 不再涨 → per-SM TMA 队列到顶, 不是 outstanding 不够
//   单 SM 打不满; 同样 kernel 大约 4 个 SM 到 ~270 GB/s 后也上不去
//     (更大 tile 的 m_sm_bw 里 8 SM TMA 能到 360)
//
//   nvcc -O3 -std=c++17 -arch=sm_120 -lcuda -o /tmp/m_tma_unroll micro_bench/m_tma_unroll.cu
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

// 8 KB tile → 99 KB smem 最多约 12 outstanding
static constexpr int TILE_M = 16;
static constexpr int TILE_N = 128;
static constexpr int TILE_ELEMS = TILE_M * TILE_N;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(float);

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

__device__ __forceinline__ void issue_one(float *sm, uint64_t desc, int tile,
                                          uint64_t *bar) {
  mbar_expect_tx(bar, TILE_BYTES);
  tma_load_2d(sm, desc, 0, tile * TILE_M, bar);
}

// -----------------------------------------------------------------------------
// serial: fire K, wait K, repeat (上次那个, 管道会抽空)
// -----------------------------------------------------------------------------
template <int K>
__global__ void tma_batch(const __grid_constant__ CUtensorMap tma, int n_tiles,
                          int iters, float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  if (threadIdx.x == 0)
    for (int s = 0; s < K; ++s)
      mbar_init(bar + s, 1);
  __syncthreads();

  int n = n_tiles - n_tiles % K;
  float acc = 0.f;
  int group = 0;
  for (int it = 0; it < iters; ++it) {
    // this CTA owns tiles blockIdx, blockIdx+grid, ...  clipped to multiple of K
    int n_mine = 0;
    for (int t = blockIdx.x; t < n; t += gridDim.x)
      ++n_mine;
    n_mine -= n_mine % K;
    for (int base = 0; base < n_mine; base += K, ++group) {
      uint32_t phase = group & 1;
      if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < K; ++i) {
          int tile = blockIdx.x + (base + i) * gridDim.x;
          issue_one(sm + i * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma), tile,
                    bar + i);
        }
      }
      __syncthreads();
      for (int i = 0; i < K; ++i)
        mbar_wait(bar + i, phase);
      acc += sm[threadIdx.x];
    }
  }
  if (acc == 1e30f)
    *sink = acc;
}

// -----------------------------------------------------------------------------
// circular: 1 个线程 unroll 预填 K, 然后 wait-oldest + issue-next 一直满管
// -----------------------------------------------------------------------------
template <int K>
__global__ void tma_circ(const __grid_constant__ CUtensorMap tma, int n_tiles,
                         int iters, float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  if (threadIdx.x == 0)
    for (int s = 0; s < K; ++s)
      mbar_init(bar + s, 1);
  __syncthreads();

  int n = n_tiles - n_tiles % K;
  int n_mine = 0;
  for (int t = blockIdx.x; t < n; t += gridDim.x)
    ++n_mine;
  n_mine -= n_mine % K;
  int total = iters * n_mine;
  float acc = 0.f;
  int issued = 0;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < K && i < n_mine; ++i)
      issue_one(sm + i * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma),
                blockIdx.x + i * gridDim.x, bar + i);
    issued = (K < n_mine ? K : n_mine);
  }
  __syncthreads();

  for (int tick = 0; tick < total; ++tick) {
    int slot = tick % K;
    uint32_t phase = (tick / K) & 1;
    mbar_wait(bar + slot, phase);
    acc += sm[slot * TILE_ELEMS + (threadIdx.x & 31)];
    __syncthreads();
    if (threadIdx.x == 0 && issued < total) {
      int local = issued % n_mine;
      int tile = blockIdx.x + local * gridDim.x;
      issue_one(sm + slot * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma), tile,
                bar + slot);
      ++issued;
    }
    __syncthreads();
  }
  if (acc == 1e30f)
    *sink = acc;
}

// -----------------------------------------------------------------------------
// K threads: 每个线程自己 issue+wait, 互不等, K 条独立 1-deep 管道并行
// -----------------------------------------------------------------------------
template <int K>
__global__ void tma_kthr(const __grid_constant__ CUtensorMap tma, int n_tiles,
                         int iters, float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  if (threadIdx.x == 0)
    for (int s = 0; s < K; ++s)
      mbar_init(bar + s, 1);
  __syncthreads();

  int n = n_tiles - n_tiles % K;
  float acc = 0.f;
  if (threadIdx.x < K) {
    int p = 0;
    for (int it = 0; it < iters; ++it) {
      for (int tile = blockIdx.x * K + threadIdx.x; tile < n;
           tile += gridDim.x * K) {
        issue_one(sm + threadIdx.x * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma),
                  tile, bar + threadIdx.x);
        mbar_wait(bar + threadIdx.x, p & 1);
        ++p;
        acc += sm[threadIdx.x * TILE_ELEMS];
      }
    }
  }
  __syncthreads();
  if (acc == 1e30f)
    *sink = acc;
}

// -----------------------------------------------------------------------------
// K warps: warp i 的 lane0 发 TMA, 跟 producer warp 一样
// -----------------------------------------------------------------------------
template <int K>
__global__ void tma_kwarp(const __grid_constant__ CUtensorMap tma, int n_tiles,
                          int iters, float *sink) {
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (threadIdx.x == 0)
    for (int s = 0; s < K; ++s)
      mbar_init(bar + s, 1);
  __syncthreads();

  int n = n_tiles - n_tiles % K;
  float acc = 0.f;
  if (warp < K && lane == 0) {
    int p = 0;
    for (int it = 0; it < iters; ++it) {
      for (int tile = blockIdx.x * K + warp; tile < n; tile += gridDim.x * K) {
        issue_one(sm + warp * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma), tile,
                  bar + warp);
        mbar_wait(bar + warp, p & 1);
        ++p;
        acc += sm[warp * TILE_ELEMS];
      }
    }
  }
  __syncthreads();
  if (acc == 1e30f)
    *sink = acc;
}

// 多 CTA 钉在 1 个 SM 上: 只有 smid==0 干活
template <int K>
__global__ void tma_kthr_sm0(const __grid_constant__ CUtensorMap tma,
                             int n_tiles, int iters, float *sink) {
  if (smid() != 0)
    return;
  extern __shared__ __align__(128) char raw[];
  float *sm = reinterpret_cast<float *>(raw);
  uint64_t *bar = reinterpret_cast<uint64_t *>(raw + K * TILE_BYTES);
  if (threadIdx.x == 0)
    for (int s = 0; s < K; ++s)
      mbar_init(bar + s, 1);
  __syncthreads();

  int n = n_tiles - n_tiles % K;
  int n_each = n / K;
  float acc = 0.f;
  if (threadIdx.x < K) {
    int p = 0;
    for (int it = 0; it < iters; ++it) {
      for (int j = 0; j < n_each; ++j) {
        int tile =
            (blockIdx.x + (threadIdx.x + j * K) * gridDim.x) % n_tiles;
        issue_one(sm + threadIdx.x * TILE_ELEMS, reinterpret_cast<uint64_t>(&tma),
                  tile, bar + threadIdx.x);
        mbar_wait(bar + threadIdx.x, p & 1);
        ++p;
        acc += sm[threadIdx.x * TILE_ELEMS];
      }
    }
  }
  __syncthreads();
  if (acc == 1e30f)
    *sink = acc;
}

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

static int smem_bytes(int k) { return k * TILE_BYTES + k * 16; }

template <int K>
static void setup() {
  int s = smem_bytes(K);
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_batch<K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, s));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_circ<K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, s));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_kthr<K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, s));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_kwarp<K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, s));
  CUDA_CHECK(cudaFuncSetAttribute((const void *)tma_kthr_sm0<K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, s));
}

int main() {
  CU_CHECK(cuInit(0));
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int SM = prop.multiProcessorCount;
  int maxsmem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &maxsmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  printf("================================================================\n");
  printf(" m_tma_unroll  %s  SM=%d  L2=%.0fMB  max smem=%d KB  tile=%d KB\n",
         prop.name, SM, prop.l2CacheSize / 1048576.0, maxsmem / 1024,
         TILE_BYTES / 1024);
  printf("================================================================\n");

  setup<1>();
  setup<2>();
  setup<4>();
  setup<8>();
  setup<12>();

  const size_t BYTES = 256ull << 20;
  const int Ncols = TILE_N;
  const int Md = (int)(BYTES / (Ncols * sizeof(float)));
  const int n_tiles = Md / TILE_M;
  float *A = nullptr, *sink = nullptr;
  CUDA_CHECK(cudaMalloc(&A, BYTES));
  CUDA_CHECK(cudaMalloc(&sink, 4));
  CUDA_CHECK(cudaMemset(A, 1, BYTES));
  CUtensorMap tA = make_tma(A, Md, Ncols);

  auto gbs = [](double bytes, float ms) { return bytes / 1e9 / (ms / 1e3); };

  printf("TMA smoke circ<4> ...\n");
  tma_circ<4><<<1, 256, smem_bytes(4)>>>(tA, n_tiles, 1, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("ok\n");

  const int iters1 = 8; // 256MB*8 = 2GB, ~30ms @ 60 GB/s, ~6ms @ 350
  const double payload = (double)BYTES * iters1;

  auto run1 = [&](const char *name, auto &&launch, int k) {
    float ms = median_ms(2, 5, launch);
    double bw = gbs(payload, ms);
    printf("  %-10s K=%-2d  %7.2f ms  %7.1f GB/s  (%4.1f KB in-flight)\n", name,
           k, ms, bw, k * TILE_BYTES / 1024.0);
    return bw;
  };

  printf("\n=== 1 CTA / 1 SM, sweep outstanding K ===\n");
  printf("  (t0 batch = 上次那种发完再等; circ = 满管; kthr/kwarp = 并行 issue)\n");

  double best = 0;
  const char *best_name = "";
  int best_k = 0;

  auto consider = [&](const char *nm, double bw, int k) {
    if (bw > best) {
      best = bw;
      best_name = nm;
      best_k = k;
    }
  };

#define ROW(K)                                                                 \
  do {                                                                         \
    int s = smem_bytes(K);                                                     \
    if (s > maxsmem)                                                           \
      break;                                                                   \
    double b;                                                                  \
    b = run1("batch",                                                          \
             [&] { tma_batch<K><<<1, 256, s>>>(tA, n_tiles, iters1, sink); },  \
             K);                                                               \
    consider("batch", b, K);                                                   \
    b = run1("circ",                                                           \
             [&] { tma_circ<K><<<1, 256, s>>>(tA, n_tiles, iters1, sink); },   \
             K);                                                               \
    consider("circ", b, K);                                                    \
    b = run1("kthr",                                                           \
             [&] { tma_kthr<K><<<1, 256, s>>>(tA, n_tiles, iters1, sink); },   \
             K);                                                               \
    consider("kthr", b, K);                                                    \
    int thr_w = std::max(256, K * 32);                                         \
    b = run1("kwarp",                                                          \
             [&] { tma_kwarp<K><<<1, thr_w, s>>>(tA, n_tiles, iters1, sink); },\
             K);                                                               \
    consider("kwarp", b, K);                                                   \
  } while (0)

  ROW(1);
  ROW(2);
  ROW(4);
  ROW(8);
  ROW(12);
#undef ROW

  printf("\n  best 1SM = %s K=%d  %.1f GB/s\n", best_name, best_k, best);

  printf("\n=== best-ish kernels, scale SM count (1 CTA/SM) ===\n");
  printf("  %6s %10s %10s %10s %10s\n", "grid", "circ/K8", "kthr/K8", "kwarp/K8",
         "kthr/K12");
  for (int g : {1, 2, 4, 6, 8, 12, SM}) {
    if (g > SM)
      continue;
    auto one = [&](auto &&fn) {
      return gbs(payload, median_ms(1, 3, fn));
    };
    double c8 = one([&] {
      tma_circ<8><<<g, 256, smem_bytes(8)>>>(tA, n_tiles, iters1, sink);
    });
    double t8 = one([&] {
      tma_kthr<8><<<g, 256, smem_bytes(8)>>>(tA, n_tiles, iters1, sink);
    });
    double w8 = one([&] {
      tma_kwarp<8><<<g, 256, smem_bytes(8)>>>(tA, n_tiles, iters1, sink);
    });
    double t12 = one([&] {
      tma_kthr<12><<<g, 256, smem_bytes(12)>>>(tA, n_tiles, iters1, sink);
    });
    printf("  %6d %10.1f %10.1f %10.1f %10.1f\n", g, c8, t8, w8, t12);
  }

  printf("\n=== 1 SM, N CTAs 并行 TMA (smid==0, kthr K=4, 32KB/CTA) ===\n");
  {
    constexpr int K = 4;
    int s = smem_bytes(K);
    int occ = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, (const void *)tma_kthr_sm0<K>, 256, s));
    printf("  occ API = %d CTA/SM with %d KB\n", occ, s / 1024);
    // launch SM*occ so every SM has `occ` CTAs; only SM0 works.
    // bytes: only SM0's CTAs actually read. Scheduler puts `occ` on SM0.
    int ncta = std::max(1, occ);
    for (int want = 1; want <= ncta; ++want) {
      int g = SM * want;
      float ms = median_ms(2, 5, [&] {
        tma_kthr_sm0<K><<<g, 256, s>>>(tA, n_tiles, iters1, sink);
      });
      double bytes = (double)want * BYTES * iters1;
      printf("  CTA/SM0=%d  grid=%d  %7.2f ms  %7.1f GB/s  (count %d CTA)\n",
             want, g, ms, gbs(bytes, ms), want);
    }
  }

  printf("\ndone. 1 SM 打满 353 GB/s 需要单 SM TMA 管道本身到屋顶;\n");
  printf("如果 K 增大带宽不再涨, 就是 per-SM TMA 吞吐/队列到顶, 不是 outstanding 不够.\n");
  return 0;
}
