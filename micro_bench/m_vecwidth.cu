// =============================================================================
// m_vecwidth — 标量访存 vs 向量化访存 (ld.f32 / ld.v2 / ld.v4)
//
// 编译:
//   nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/m_vecwidth micro_bench/m_vecwidth.cu
//
// 测的是同一份字节、同一套 coalesced 顺序流，只换每条 LD/ST 的位宽。
// PTX 锁死宽度，避免 nvcc 把 4 次标量 load 自动收成 LDG.128。
// =============================================================================

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e) {                                                                   \
      printf("err %d %s\n", __LINE__, cudaGetErrorString(e));                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- PTX load/store: lock issue width ---------------------------------------
__device__ __forceinline__ float ld32(const float *p) {
  float v;
  asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st32(float *p, float v) {
  asm volatile("st.global.cs.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ float2 ld64(const float2 *p) {
  float2 v;
  asm volatile("ld.global.cs.v2.f32 {%0,%1}, [%2];"
               : "=f"(v.x), "=f"(v.y)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void st64(float2 *p, float2 v) {
  asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};" ::"l"(p), "f"(v.x), "f"(v.y)
               : "memory");
}
__device__ __forceinline__ float4 ld128(const float4 *p) {
  float4 v;
  asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void st128(float4 *p, float4 v) {
  asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x),
               "f"(v.y), "f"(v.z), "f"(v.w)
               : "memory");
}

// ---- A. grid-stride STREAM copy：每线程每次搬 4/8/16B，覆盖同一 n floats ----
__global__ void copy32(const float *__restrict__ in, float *__restrict__ out,
                       size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s)
    st32(out + i, ld32(in + i));
}
__global__ void copy64(const float2 *__restrict__ in, float2 *__restrict__ out,
                       size_t n2) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n2; i += s)
    st64(out + i, ld64(in + i));
}
__global__ void copy128(const float4 *__restrict__ in, float4 *__restrict__ out,
                        size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s)
    st128(out + i, ld128(in + i));
}

// ---- B. load-only / store-only ----------------------------------------------
__global__ void ldonly32(const float *__restrict__ in, float *sink, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += s)
    acc += ld32(in + i);
  if (acc == 1e30f)
    *sink = acc;
}
__global__ void ldonly128(const float4 *__restrict__ in, float *sink,
                          size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += s) {
    float4 v = ld128(in + i);
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1e30f)
    *sink = acc;
}
__global__ void stonly32(float *__restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s)
    st32(out + i, (float)(i & 255));
}
__global__ void stonly128(float4 *__restrict__ out, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s)
    st128(out + i, make_float4(1.f, 2.f, 3.f, 4.f));
}

// ---- C. 每线程搬 16B：4×标量(步长4，合并差) vs 1×v4(合并好) ---------------
// thread t 读 in[4t], in[4t+1], in[4t+2], in[4t+3]
// 同一 warp 第一次 ld32 的地址是 0,16,32,... → 32B sector 里只有 4B 有效
__global__ void copy4x32_stride4(const float *__restrict__ in,
                                 float *__restrict__ out, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s) {
    const float *p = in + i * 4;
    float *q = out + i * 4;
    float v0 = ld32(p + 0), v1 = ld32(p + 1), v2 = ld32(p + 2), v3 = ld32(p + 3);
    st32(q + 0, v0);
    st32(q + 1, v1);
    st32(q + 2, v2);
    st32(q + 3, v3);
  }
}
__global__ void copy1x128_pack(const float4 *__restrict__ in,
                               float4 *__restrict__ out, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s)
    st128(out + i, ld128(in + i));
}

// ---- D. elementwise add：2 读 1 写 ------------------------------------------
__global__ void add32(const float *__restrict__ a, const float *__restrict__ b,
                      float *__restrict__ c, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s)
    st32(c + i, ld32(a + i) + ld32(b + i));
}
__global__ void add128(const float4 *__restrict__ a, const float4 *__restrict__ b,
                       float4 *__restrict__ c, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n4; i += s) {
    float4 x = ld128(a + i), y = ld128(b + i);
    st128(c + i, make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w));
  }
}

// ---- timing -----------------------------------------------------------------
template <typename F> static float bench_ms(F &&fn, int warm = 3, int rep = 10) {
  for (int i = 0; i < warm; ++i)
    fn();
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for (int i = 0; i < rep; ++i)
    fn();
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return ms / (float)rep;
}

static void row(const char *name, double bytes, float ms, double ref_gbps) {
  double gbps = bytes / 1e9 / (ms * 1e-3);
  printf("  %-28s %8.3f ms  %7.1f GB/s", name, ms, gbps);
  if (ref_gbps > 0)
    printf("   %5.2fx vs scalar", gbps / ref_gbps);
  printf("\n");
}

int main() {
  CK(cudaSetDevice(0));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int SM = p.multiProcessorCount;
  int mem_clk_khz = 0;
  CK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0));
  // GDDR 粗估: 2 * memClk_Hz * busBytes
  double theo =
      (double)mem_clk_khz * 1e3 * 2.0 * (p.memoryBusWidth / 8.0) / 1e9;
  printf("%s  sm_%d%d  SM=%d  L2=%.0fMB  bus=%dbit  memclk=%.0fMHz  theo~%.0f GB/s\n",
         p.name, p.major, p.minor, SM, p.l2CacheSize / 1e6, p.memoryBusWidth,
         mem_clk_khz / 1000.0, theo);

  const size_t N = 64ull << 20; // 64M floats = 256MB  (>L2)
  const size_t BYTES = N * sizeof(float);
  float *A, *B, *C, *sink;
  CK(cudaMalloc(&A, BYTES));
  CK(cudaMalloc(&B, BYTES));
  CK(cudaMalloc(&C, BYTES));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(A, 1, BYTES));
  CK(cudaMemset(B, 2, BYTES));

  const int BLK = 256;
  auto run = [&](int cta_per_sm, const char *tag) {
    int grid = SM * cta_per_sm;
    printf("\n== %s  grid=%d (%d CTA/SM × %d thr)  working set=%.0fMB ==\n", tag,
           grid, cta_per_sm, BLK, BYTES / 1e6);

    // A. STREAM copy (read+write → 2×BYTES)
    printf("-- STREAM copy (R+W, 2×bytes) --\n");
    float t32 = bench_ms([&] { copy32<<<grid, BLK>>>(A, C, N); });
    double s32 = (2.0 * BYTES) / 1e9 / (t32 * 1e-3);
    row("copy  ld.f32  / st.f32", 2.0 * BYTES, t32, 0);
    float t64 = bench_ms([&] {
      copy64<<<grid, BLK>>>((float2 *)A, (float2 *)C, N / 2);
    });
    row("copy  ld.v2   / st.v2 ", 2.0 * BYTES, t64, s32);
    float t128 = bench_ms([&] {
      copy128<<<grid, BLK>>>((float4 *)A, (float4 *)C, N / 4);
    });
    row("copy  ld.v4   / st.v4 ", 2.0 * BYTES, t128, s32);

    // B. load / store only
    printf("-- load-only / store-only --\n");
    float lr32 = bench_ms([&] { ldonly32<<<grid, BLK>>>(A, sink, N); });
    double lr32b = BYTES / 1e9 / (lr32 * 1e-3);
    row("load  ld.f32", BYTES, lr32, 0);
    float lr128 = bench_ms([&] {
      ldonly128<<<grid, BLK>>>((float4 *)A, sink, N / 4);
    });
    row("load  ld.v4", BYTES, lr128, lr32b);
    float sw32 = bench_ms([&] { stonly32<<<grid, BLK>>>(C, N); });
    double sw32b = BYTES / 1e9 / (sw32 * 1e-3);
    row("store st.f32", BYTES, sw32, 0);
    float sw128 = bench_ms([&] { stonly128<<<grid, BLK>>>((float4 *)C, N / 4); });
    row("store st.v4", BYTES, sw128, sw32b);

    // C. packed 16B / thread
    printf("-- per-thread 16B: 4×ld.f32 stride-4 vs 1×ld.v4 --\n");
    float bad = bench_ms([&] { copy4x32_stride4<<<grid, BLK>>>(A, C, N / 4); });
    double badb = (2.0 * BYTES) / 1e9 / (bad * 1e-3);
    row("4x ld.f32  (stride-4)", 2.0 * BYTES, bad, 0);
    float pack = bench_ms([&] {
      copy1x128_pack<<<grid, BLK>>>((float4 *)A, (float4 *)C, N / 4);
    });
    row("1x ld.v4   (packed)  ", 2.0 * BYTES, pack, badb);

    // D. add
    printf("-- elementwise add (2R+1W) --\n");
    float a32 = bench_ms([&] { add32<<<grid, BLK>>>(A, B, C, N); });
    double a32b = (3.0 * BYTES) / 1e9 / (a32 * 1e-3);
    row("add   scalar f32", 3.0 * BYTES, a32, 0);
    float a128 = bench_ms([&] {
      add128<<<grid, BLK>>>((float4 *)A, (float4 *)B, (float4 *)C, N / 4);
    });
    row("add   float4    ", 3.0 * BYTES, a128, a32b);
  };

  run(1, "low occupancy (LSU issue bound)");
  run(6, "high occupancy (DRAM bound)");

  printf("\n== STREAM copy occupancy sweep (same 256MB, R+W) ==\n");
  printf("  %-8s %10s %10s %10s %8s\n", "CTA/SM", "f32 GB/s", "v2 GB/s", "v4 GB/s",
         "v4/f32");
  int sweep[] = {1, 2, 3, 4, 6, 8};
  for (int k = 0; k < 6; ++k) {
    int cta = sweep[k];
    int grid = SM * cta;
    float t32 = bench_ms([&] { copy32<<<grid, BLK>>>(A, C, N); });
    float t64 = bench_ms([&] {
      copy64<<<grid, BLK>>>((float2 *)A, (float2 *)C, N / 2);
    });
    float t128 = bench_ms([&] {
      copy128<<<grid, BLK>>>((float4 *)A, (float4 *)C, N / 4);
    });
    double b32 = (2.0 * BYTES) / 1e9 / (t32 * 1e-3);
    double b64 = (2.0 * BYTES) / 1e9 / (t64 * 1e-3);
    double b128 = (2.0 * BYTES) / 1e9 / (t128 * 1e-3);
    printf("  %-8d %10.1f %10.1f %10.1f %7.2fx\n", cta, b32, b64, b128,
           b128 / b32);
  }
  return 0;
}
