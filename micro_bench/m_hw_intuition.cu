// Interview-oriented extras: cache-line/sector, Little's Law (BDP),
// latency hierarchy, wave quantization, clock throttle.
// Build: nvcc -O3 -arch=sm_120 -o m_hw_intuition m_hw_intuition.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e=(x);if(e){printf("err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)

static int sm_clk_khz() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0);
  return v;
}

__device__ __forceinline__ unsigned ld_ca(const unsigned* p) {
  unsigned v;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ld_cg(const unsigned* p) {
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ unsigned ld_cv(const unsigned* p) {
  unsigned v;
  asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__global__ void k_chase(const unsigned* a, int hops, int start, int mod,
                        unsigned long long* cyc, unsigned* sink, int kind) {
  unsigned p = (unsigned)start;
  unsigned long long t0 = clock64();
  if (kind == 0) {
    for (int i = 0; i < hops; ++i) p = ld_ca(a + (p & (unsigned)mod));
  } else if (kind == 1) {
    for (int i = 0; i < hops; ++i) p = ld_cg(a + (p & (unsigned)mod));
  } else {
    for (int i = 0; i < hops; ++i) p = ld_cv(a + (p & (unsigned)mod));
  }
  unsigned long long t1 = clock64();
  *cyc = t1 - t0;
  *sink = p;
}

// Streaming bandwidth, working-set masked. 16B vector loads.
__global__ void k_ws(const float4* g, unsigned mask, float* out) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = gridDim.x * blockDim.x;
  float s = 0;
#pragma unroll 4
  for (int k = 0; k < 256; ++k) {
    float4 v = g[i & mask];
    s += v.x + v.y + v.z + v.w;
    i += step;
  }
  if (s == 1234.5f) out[0] = s;
}

// Warp lanes load 4B at lane * lane_stride. Distinguishes 32B sector vs 128B line
// at lane_stride = 64 and 128 (see comments in main).
__global__ void k_lane_stride(const char* p, size_t nbytes, int lane_stride, float* out) {
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int nwarps = (gridDim.x * blockDim.x) >> 5;
  size_t span = (size_t)32 * (size_t)lane_stride;
  size_t off = (size_t)warp * span;
  size_t step = (size_t)nwarps * span;
  float s = 0;
  int n = 0;
  for (; off + (size_t)lane * (size_t)lane_stride + 4 <= nbytes; off += step) {
    s += *reinterpret_cast<const float*>(p + off + (size_t)lane * (size_t)lane_stride);
    ++n;
  }
  if (s == 1e30f) out[0] = s + (float)n;
}

template <int K>
__global__ void k_mlp(const float4* in, size_t n, float* out) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0, 0, 0, 0);
  for (size_t i = tid; i + (size_t)(K - 1) * grid < n; i += grid * K) {
    float4 v[K];
#pragma unroll
    for (int k = 0; k < K; ++k) v[k] = in[i + (size_t)k * grid];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      acc.x += v[k].x;
      acc.y += v[k].y;
      acc.z += v[k].z;
      acc.w += v[k].w;
    }
  }
  if (acc.x == 1e30f) out[threadIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

__global__ void k_ffma_dep(float* out, int it) {
  float x = threadIdx.x * 1e-3f + 1.f, y = x + 1, z = x + 2, w = x + 3;
  for (int i = 0; i < it; ++i) {
    x = fmaf(x, y, z);
    y = fmaf(y, z, w);
    z = fmaf(z, w, x);
    w = fmaf(w, x, y);
  }
  if (x + y + z + w == 1e30f) out[0] = x;
}

__global__ void k_ffma_time(float* out, int it, unsigned long long* cyc) {
  float x = 1.001f;
  unsigned long long t0 = clock64();
  for (int i = 0; i < it; ++i) x = fmaf(x, 1.000001f, 0.5f);
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) *cyc = t1 - t0;
  if (x == 1e30f) out[0] = x;
}

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int sm = p.multiProcessorCount;
  int clk = sm_clk_khz();
  int bus = 0, mclk = 0;
  cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
  cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0);
  printf("=== device ===\n");
  printf("%s  sm_%d%d  SMs=%d  L2=%.1f MB  clock=%.2f GHz  bus=%d-bit  memClk=%.0f MHz\n",
         p.name, p.major, p.minor, sm, p.l2CacheSize / 1048576.0, clk / 1e6, bus, mclk / 1000.0);
  printf("shared/SM=%zu KB  optin=%d  maxThr/SM=%d  regs/SM=%d\n",
         p.sharedMemPerMultiprocessor / 1024, 0, p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor);
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  printf("shared opt-in %d B\n\n", optin);

  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));
  float* sink;
  CK(cudaMalloc(&sink, 4096));

  // ----- 1. pointer-chase latency -----
  printf("=== 1. pointer-chase latency (cyc/hop, 200k hops) ===\n");
  printf("kind  WS        cyc/hop    ns\n");
  const int hops = 200000;
  unsigned long long* d_cyc;
  unsigned* d_tail;
  CK(cudaMalloc(&d_cyc, 8));
  CK(cudaMalloc(&d_tail, 4));
  struct {
    const char* name;
    size_t bytes;
    int kind;
  } lat_cases[] = {
      {"ca ", 8ull << 10, 0},     {"ca ", 128ull << 10, 0}, {"ca ", 1ull << 20, 0},
      {"ca ", 4ull << 20, 0},     {"ca ", 16ull << 20, 0},  {"ca ", 64ull << 20, 0},
      {"ca ", 256ull << 20, 0},   {"cg ", 8ull << 10, 1},   {"cg ", 128ull << 10, 1},
      {"cg ", 4ull << 20, 1},     {"cg ", 16ull << 20, 1},  {"cg ", 64ull << 20, 1},
      {"cg ", 256ull << 20, 1},   {"cv ", 8ull << 10, 2},   {"cv ", 4ull << 20, 2},
      {"cv ", 64ull << 20, 2},    {"cv ", 256ull << 20, 2},
  };
  unsigned* d_link = nullptr;
  size_t link_cap = 256ull << 20;
  CK(cudaMalloc(&d_link, link_cap));
  unsigned* h_link = (unsigned*)malloc(link_cap);
  if (!h_link) {
    printf("host alloc fail\n");
    return 1;
  }
  // 128 B hops so each chase misses the previous line
  const unsigned step = 32;
  auto fill = [&](size_t bytes) {
    unsigned n = (unsigned)(bytes / 4);
    unsigned mod = n - 1;
    for (unsigned i = 0; i < n; ++i) h_link[i] = (i + step) & mod;
    CK(cudaMemcpy(d_link, h_link, bytes, cudaMemcpyHostToDevice));
    return 0;
  };
  for (auto& c : lat_cases) {
    if (fill(c.bytes)) return 1;
    unsigned n = (unsigned)(c.bytes / 4);
    unsigned mod = n - 1;
    k_chase<<<1, 1>>>(d_link, hops, 0, (int)mod, d_cyc, d_tail, c.kind);
    CK(cudaDeviceSynchronize());
    k_chase<<<1, 1>>>(d_link, hops, 0, (int)mod, d_cyc, d_tail, c.kind);
    CK(cudaDeviceSynchronize());
    unsigned long long cy = 0;
    CK(cudaMemcpy(&cy, d_cyc, 8, cudaMemcpyDeviceToHost));
    double cph = (double)cy / hops;
    printf("%s  %7.1f KB  %7.1f  %6.1f\n", c.name, c.bytes / 1024.0, cph, cph * 1e6 / clk);
  }
  printf("\n");

  // ----- 2. L2 capacity cliff -----
  printf("=== 2. working-set bandwidth cliff (float4 streaming) ===\n");
  size_t big = 512ull << 20;
  float4* d_ws;
  CK(cudaMalloc(&d_ws, big));
  CK(cudaMemset(d_ws, 0, big));
  int thr = 256, blk = sm * 8;
  double acc = (double)thr * blk * 256 * 16;
  printf(" WS          GB/s\n");
  for (int e = 14; e <= 29; ++e) {
    unsigned n = (1u << e) / 16, mask = n - 1;
    k_ws<<<blk, thr>>>(d_ws, mask, sink);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ev0));
    for (int r = 0; r < 5; ++r) k_ws<<<blk, thr>>>(d_ws, mask, sink);
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    printf("%8d KB  %7.1f\n", (1 << e) / 1024, acc * 5 / 1e9 / (elapsed_ms(ev0, ev1) / 1000.0));
  }
  printf("\n");

  // ----- 3. sector vs line -----
  printf("=== 3. lane-stride (useful GB/s). 32B sector vs 128B line diverge at stride 64/128 ===\n");
  printf("predicted DRAM bytes/warp: stride64 sector=1024 line=2048; stride128 sector=1024 line=4096\n");
  size_t sbytes = 256ull << 20;
  char* d_st;
  CK(cudaMalloc(&d_st, sbytes));
  CK(cudaMemset(d_st, 1, sbytes));
  int sthr = 256, sblk = sm * 16;
  printf(" strideB   usefulGB/s   vs4B   implied_fetchB  (4 * peak/this)\n");
  double peak_stride = 0;
  int strides[] = {4, 8, 16, 32, 64, 128, 256};
  for (int st : strides) {
    k_lane_stride<<<sblk, sthr>>>(d_st, sbytes, st, sink);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ev0));
    for (int r = 0; r < 4; ++r) k_lane_stride<<<sblk, sthr>>>(d_st, sbytes, st, sink);
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    int nwarps = (sblk * sthr) >> 5;
    size_t span = (size_t)32 * st;
    size_t iters = sbytes / (span * (size_t)nwarps);
    double useful = (double)nwarps * iters * 128.0 * 4;  // 32 lanes * 4B * iters * repeats
    double gbs = useful / 1e9 / (elapsed_ms(ev0, ev1) / 1000.0);
    if (st == 4) peak_stride = gbs;
    double implied = (peak_stride > 0) ? 4.0 * peak_stride / gbs : 0;
    printf("%8d  %10.1f  %5.2fx  %8.1f\n", st, gbs, peak_stride / gbs, implied);
  }
  printf("\n");

  // ----- 4. Little's Law: occupancy x MLP -----
  printf("=== 4. HBM/GDDR Little's Law: warps/SM x independent float4 loads ===\n");
  size_t n4 = (256ull << 20) / 16;
  float4* d_mlp;
  CK(cudaMalloc(&d_mlp, n4 * 16));
  CK(cudaMemset(d_mlp, 0, n4 * 16));
  printf("K warps/SM  GB/s   in-flight_est_KB  (threads*K*16B)\n");
  auto run_mlp = [&](int K, int warps_per_sm) {
    int t = 128;
    int wpb = t / 32;
    int bpsm = warps_per_sm / wpb;
    if (bpsm < 1) {
      t = warps_per_sm * 32;
      wpb = warps_per_sm;
      bpsm = 1;
    }
    int grid = bpsm * sm;
    auto launch = [&]() {
      if (K == 1) k_mlp<1><<<grid, t>>>(d_mlp, n4, sink);
      else if (K == 2) k_mlp<2><<<grid, t>>>(d_mlp, n4, sink);
      else if (K == 4) k_mlp<4><<<grid, t>>>(d_mlp, n4, sink);
      else k_mlp<8><<<grid, t>>>(d_mlp, n4, sink);
    };
    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ev0));
    launch();
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    double bytes = (double)n4 * 16;
    double gbs = bytes / 1e9 / (elapsed_ms(ev0, ev1) / 1000.0);
    double inflight = (double)grid * t * K * 16.0 / 1024.0;
    printf("%d %8d  %6.1f  %10.1f\n", K, warps_per_sm, gbs, inflight);
    return 0;
  };
  int ks[] = {1, 2, 4, 8};
  int wps[] = {1, 2, 4, 8, 16, 24, 32, 48};
  for (int K : ks) {
    for (int w : wps) {
      if (w * 32 > p.maxThreadsPerMultiProcessor) continue;
      if (run_mlp(K, w)) return 1;
    }
  }
  printf("\n");

  // ----- 5. wave quantization -----
  printf("=== 5. wave quantization (1 block/SM via smem cap, 256 thr) ===\n");
  CK(cudaFuncSetAttribute(k_ffma_dep, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
  // occupancy 1 block/SM: request almost all shared
  int smem = (int)p.sharedMemPerMultiprocessor - 2048;
  if (smem > optin) smem = optin;
  if (smem < 0) smem = 0;
  int it = 8000;
  printf("blocks  waves     ms   eff%%\n");
  double base = 0;
  int gs[] = {sm, sm + 1, 2 * sm, 2 * sm + 1, 4 * sm, 4 * sm + 1, 8 * sm, 8 * sm + 1};
  for (int g : gs) {
    k_ffma_dep<<<g, 256, smem>>>(sink, it);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(ev0));
    k_ffma_dep<<<g, 256, smem>>>(sink, it);
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    float ms = elapsed_ms(ev0, ev1);
    if (g == sm) base = ms;
    printf("%6d  %5.2f  %6.2f  %5.0f\n", g, (double)g / sm, ms, 100.0 * (base * g / sm) / ms);
  }
  printf("\n");

  // ----- 6. FFMA dep latency + implied SM clock vs wall -----
  printf("=== 6. FFMA dep latency and implied SM clock ===\n");
  unsigned long long* d_fcyc;
  CK(cudaMalloc(&d_fcyc, 8));
  int fit = 2000000;
  k_ffma_time<<<1, 1>>>(sink, fit, d_fcyc);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(ev0));
  k_ffma_time<<<1, 1>>>(sink, fit, d_fcyc);
  CK(cudaEventRecord(ev1));
  CK(cudaEventSynchronize(ev1));
  unsigned long long fcy = 0;
  CK(cudaMemcpy(&fcy, d_fcyc, 8, cudaMemcpyDeviceToHost));
  double wall_ms = elapsed_ms(ev0, ev1);
  double cpi = (double)fcy / fit;
  double impl_ghz = (fcy / 1e6) / wall_ms;
  printf("cyc/FFMA=%.2f  wall=%.3f ms  implied SM clock=%.2f GHz  (prop %.2f GHz)\n",
         cpi, wall_ms, impl_ghz, clk / 1e6);

  printf("\n=== 6b. 8s compute, sample implied clock each second ===\n");
  int chunk = 8000000;
  for (int s = 0; s < 8; ++s) {
    CK(cudaEventRecord(ev0));
    k_ffma_dep<<<sm, 256>>>(sink, chunk);
    CK(cudaEventRecord(ev1));
    CK(cudaEventSynchronize(ev1));
    printf("  t+%ds  %6.1f ms\n", s, elapsed_ms(ev0, ev1));
  }

  printf("\ndone.\n");
  return 0;
}
