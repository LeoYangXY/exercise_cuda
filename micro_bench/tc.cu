#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>

// RTX 50 (sm_120) mma.sync m16n8k16.f32.f16.f16.f32
// 实验目的：把 TC 的 throughput / latency 测干净，再用这两个数分析
// 「mma 之后立刻做 epilogue」时，软件 ILP（独立寄存器 + 分组发射）
// 能不能和同 SMSP 上的 TLP 打平。
//
// 关键约束：warp 是 in-order issue。寄存器独立只是必要不充分——
// 如果指令流是 mma; fadd; mma; fadd，碰到 fadd 的 scoreboard 就会停，
// 后面那条独立 mma 发不出去。TLP 是硬件换 warp 跳过这条停顿。

#ifndef NMMA
#define NMMA 65536
#endif
#ifndef NREP
#define NREP 7
#endif

__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3,
                                    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// volatile PTX，防止编译器把 fadd 挪到一串 mma 后面（那样 naive 会偷偷变成 ILP）
__device__ __forceinline__ void fadd(float& acc, float x) {
  asm volatile("add.f32 %0, %0, %1;" : "+f"(acc) : "f"(x));
}

__device__ __forceinline__ long long clk() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ long long clk_dep(float dep) {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) : "f"(dep));
  return t;
}

struct Frag {
  unsigned a0, a1, a2, a3, b0, b1;
};
__device__ __forceinline__ Frag load_ab(const unsigned* A, const unsigned* B, int lane) {
  Frag f;
  f.a0 = A[lane * 4];
  f.a1 = A[lane * 4 + 1];
  f.a2 = A[lane * 4 + 2];
  f.a3 = A[lane * 4 + 3];
  f.b0 = B[lane * 2];
  f.b1 = B[lane * 2 + 1];
  return f;
}

// ---------- 1. 纯 MMA：latency = 1 条依赖链 ----------
__global__ void kLat(unsigned* A, unsigned* B, float* O, long long* C) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / 8; i++) {
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
  }
  long long t1 = clk_dep(d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[0] = t1 - t0;
    O[0] = d0 + d1 + d2 + d3;
  }
}

// ---------- 1b. 纯 MMA：throughput = C 条独立累加器 ----------
template <int CH>
__global__ void kTput(unsigned* A, unsigned* B, float* O, long long* Cclk) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  Frag f = load_ab(A, B, lane);
  float d[CH][4];
#pragma unroll
  for (int c = 0; c < CH; c++) d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0.f;
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / CH; i++) {
#pragma unroll
    for (int c = 0; c < CH; c++)
      mma(d[c][0], d[c][1], d[c][2], d[c][3], f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  long long t1 = clk_dep(s);
  if (lane == 0) {
    Cclk[0] = t1 - t0;
    O[0] = s;
  }
}

// ---------- FADD lat / tput（给理论模型用）----------
__global__ void kFaddLat(float* O, long long* C) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  float x = 1.000001f + lane * 1e-4f, a = 0.5f;
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / 8; i++) {
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
    fadd(x, a);
  }
  long long t1 = clk_dep(x);
  if (lane == 0) {
    C[0] = t1 - t0;
    O[0] = x;
  }
}

template <int CH>
__global__ void kFaddTput(float* O, long long* Cclk) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  float x[CH];
#pragma unroll
  for (int c = 0; c < CH; c++) x[c] = 1.f + c + lane * 1e-4f;
  float a = 0.500001f;
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / CH; i++) {
#pragma unroll
    for (int c = 0; c < CH; c++) fadd(x[c], a);
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) s += x[c];
  long long t1 = clk_dep(s);
  if (lane == 0) {
    Cclk[0] = t1 - t0;
    O[0] = s;
  }
}

// ---------- 2. epilogue：1 链串行 MMA→FADD（下界，latency 暴露）----------
__global__ void kEpi1(unsigned* A, unsigned* B, float* O, long long* C) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, acc = 0;
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA; i++) {
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    fadd(acc, d0);
  }
  long long t1 = clk_dep(acc + d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[0] = t1 - t0;
    O[0] = acc;
  }
}

// 交错：mma; fadd; mma; fadd; …  4 组独立寄存器，但 in-order 会被 fadd 卡住
template <int CH>
__global__ void kEpiInterleave(unsigned* A, unsigned* B, float* O, long long* Cclk) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  Frag f = load_ab(A, B, lane);
  float d[CH][4], acc[CH];
#pragma unroll
  for (int c = 0; c < CH; c++) {
    d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0.f;
    acc[c] = 0.f;
  }
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / CH; i++) {
#pragma unroll
    for (int c = 0; c < CH; c++) {
      mma(d[c][0], d[c][1], d[c][2], d[c][3], f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
      fadd(acc[c], d[c][0]);
    }
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) s += acc[c];
  long long t1 = clk_dep(s);
  if (lane == 0) {
    Cclk[0] = t1 - t0;
    O[0] = s;
  }
}

// 分组：mma×CH; fadd×CH  —— 软件把独立 mma 连续发出去
template <int CH>
__global__ void kEpiGrouped(unsigned* A, unsigned* B, float* O, long long* Cclk) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp != 0) return;
  Frag f = load_ab(A, B, lane);
  float d[CH][4], acc[CH];
#pragma unroll
  for (int c = 0; c < CH; c++) {
    d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0.f;
    acc[c] = 0.f;
  }
  __syncwarp();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / CH; i++) {
#pragma unroll
    for (int c = 0; c < CH; c++)
      mma(d[c][0], d[c][1], d[c][2], d[c][3], f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
#pragma unroll
    for (int c = 0; c < CH; c++) fadd(acc[c], d[c][0]);
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) s += acc[c];
  long long t1 = clk_dep(s);
  if (lane == 0) {
    Cclk[0] = t1 - t0;
    O[0] = s;
  }
}

// TLP：nkeep 个 warp 挤在同一个 SMSP（warp id = 0,4,8,...）
// 每个 warp 只有 1 条链，指令流就是 mma; fadd。靠调度器换 warp 填洞。
__global__ void kEpiTLP_same(unsigned* A, unsigned* B, float* O, long long* C, int nkeep) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((warp & 3) != 0) return;
  int id = warp >> 2;
  if (id >= nkeep) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, acc = 0;
  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / nkeep; i++) {
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    fadd(acc, d0);
  }
  long long t1 = clk_dep(acc + d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[warp] = t1 - t0;
    O[warp] = acc;
  }
}

// 对照：nkeep 个 warp 铺在不同 SMSP（warp 0,1,2,...）——用的是多套 TC，不是公平对比
__global__ void kEpiTLP_spread(unsigned* A, unsigned* B, float* O, long long* C, int nkeep) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp >= nkeep) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0, acc = 0;
  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / nkeep; i++) {
    mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
    fadd(acc, d0);
  }
  long long t1 = clk_dep(acc + d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[warp] = t1 - t0;
    O[warp] = acc;
  }
}

// 纯 MMA 的 TLP（无 epilogue），铺开到不同 SMSP
__global__ void kTputTLP_spread(unsigned* A, unsigned* B, float* O, long long* C, int nkeep) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp >= nkeep) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / nkeep; i++) mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
  long long t1 = clk_dep(d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[warp] = t1 - t0;
    O[warp] = d0;
  }
}

__global__ void kTputTLP_same(unsigned* A, unsigned* B, float* O, long long* C, int nkeep) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((warp & 3) != 0) return;
  int id = warp >> 2;
  if (id >= nkeep) return;
  Frag f = load_ab(A, B, lane);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < NMMA / nkeep; i++) mma(d0, d1, d2, d3, f.a0, f.a1, f.a2, f.a3, f.b0, f.b1);
  long long t1 = clk_dep(d0 + d1 + d2 + d3);
  if (lane == 0) {
    C[warp] = t1 - t0;
    O[warp] = d0;
  }
}

static long long median(long long* v, int n) {
  std::sort(v, v + n);
  return v[n / 2];
}

static void ck(const char* what) {
  cudaError_t e = cudaGetLastError();
  if (e) printf("CUDA err after %s: %s\n", what, cudaGetErrorString(e));
}

template <typename K>
static long long bench1(K k, unsigned* dA, unsigned* dB, float* dO, long long* dC) {
  // 单 warp kernel：必须 32 thread。1024 thread 会按 block 分配寄存器，
  // ILP=16（76+ regs）会超过 64K/SM，kernel 直接没 launch，clock 读到 0。
  long long h[NREP];
  k<<<1, 32>>>(dA, dB, dO, dC);
  cudaDeviceSynchronize();
  ck("warmup");
  for (int r = 0; r < NREP; r++) {
    cudaMemset(dC, 0, 256);
    k<<<1, 32>>>(dA, dB, dO, dC);
    cudaDeviceSynchronize();
    cudaMemcpy(&h[r], dC, 8, cudaMemcpyDeviceToHost);
  }
  return median(h, NREP);
}

template <typename K>
static long long bench1f(K k, float* dO, long long* dC) {
  long long h[NREP];
  k<<<1, 32>>>(dO, dC);
  cudaDeviceSynchronize();
  for (int r = 0; r < NREP; r++) {
    cudaMemset(dC, 0, 256);
    k<<<1, 32>>>(dO, dC);
    cudaDeviceSynchronize();
    cudaMemcpy(&h[r], dC, 8, cudaMemcpyDeviceToHost);
  }
  return median(h, NREP);
}

static long long bench_tlp(void (*k)(unsigned*, unsigned*, float*, long long*, int),
                           unsigned* dA, unsigned* dB, float* dO, long long* dC, int nkeep) {
  long long h[NREP];
  k<<<1, 1024>>>(dA, dB, dO, dC, nkeep);
  cudaDeviceSynchronize();
  for (int r = 0; r < NREP; r++) {
    cudaMemset(dC, 0, 256);
    k<<<1, 1024>>>(dA, dB, dO, dC, nkeep);
    cudaDeviceSynchronize();
    long long tmp[32] = {0};
    cudaMemcpy(tmp, dC, 256, cudaMemcpyDeviceToHost);
    long long mx = 0;
    for (int w = 0; w < 32; w++)
      if (tmp[w] > mx) mx = tmp[w];
    h[r] = mx;
  }
  return median(h, NREP);
}

static void pr(const char* name, long long cyc, double n_mma) {
  printf("  %-22s  wall=%10lld   cyc/mma=%7.2f\n", name, cyc, (double)cyc / n_mma);
}

int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int clk_khz = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  printf("GPU  %s   cc=%d.%d   SM=%d   clk=%.2f GHz\n", p.name, p.major, p.minor,
         p.multiProcessorCount, clk_khz / 1e6);
  printf("MMA  mma.sync.m16n8k16.row.col.f32.f16.f16.f32   NMMA=%d  median of %d\n\n", NMMA,
         NREP);

  unsigned hA[128], hB[64];
  for (int i = 0; i < 128; i++) hA[i] = 0x3c003c00;
  for (int i = 0; i < 64; i++) hB[i] = 0x3c003c00;
  unsigned *dA, *dB;
  float* dO;
  long long* dC;
  cudaMalloc(&dA, 512);
  cudaMalloc(&dB, 256);
  cudaMalloc(&dO, 1024);
  cudaMalloc(&dC, 256);
  cudaMemcpy(dA, hA, 512, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);

  printf("== 1. pure MMA  latency / throughput (1 warp) ==\n");
  long long lat = bench1(kLat, dA, dB, dO, dC);
  pr("LAT  (1 chain)", lat, NMMA);
  long long tp1 = bench1(kTput<1>, dA, dB, dO, dC);
  long long tp2 = bench1(kTput<2>, dA, dB, dO, dC);
  long long tp4 = bench1(kTput<4>, dA, dB, dO, dC);
  long long tp8 = bench1(kTput<8>, dA, dB, dO, dC);
  long long tp16 = bench1(kTput<16>, dA, dB, dO, dC);
  pr("TPUT ILP=1", tp1, NMMA);
  pr("TPUT ILP=2", tp2, NMMA);
  pr("TPUT ILP=4", tp4, NMMA);
  pr("TPUT ILP=8", tp8, NMMA);
  pr("TPUT ILP=16", tp16, NMMA);

  printf("\n== 1b. pure MMA TLP same-SMSP (warps 0,4,8,...) vs spread SMSP ==\n");
  for (int w : {1, 2, 4, 8}) {
    long long t = bench_tlp(kTputTLP_same, dA, dB, dO, dC, w);
    char buf[64];
    snprintf(buf, sizeof(buf), "TPUT TLP=%d same", w);
    pr(buf, t, NMMA);
  }
  for (int w : {1, 2, 4}) {
    long long t = bench_tlp(kTputTLP_spread, dA, dB, dO, dC, w);
    char buf[64];
    snprintf(buf, sizeof(buf), "TPUT TLP=%d spread", w);
    pr(buf, t, NMMA);
  }

  printf("\n== 1c. FADD lat / tput (1 warp) ==\n");
  long long fl = bench1f(kFaddLat, dO, dC);
  long long ft1 = bench1f(kFaddTput<1>, dO, dC);
  long long ft4 = bench1f(kFaddTput<4>, dO, dC);
  long long ft8 = bench1f(kFaddTput<8>, dO, dC);
  long long ft16 = bench1f(kFaddTput<16>, dO, dC);
  printf("  %-22s  wall=%10lld   cyc/fadd=%7.2f\n", "FADD LAT", fl, (double)fl / NMMA);
  printf("  %-22s  wall=%10lld   cyc/fadd=%7.2f\n", "FADD TPUT ILP=1", ft1, (double)ft1 / NMMA);
  printf("  %-22s  wall=%10lld   cyc/fadd=%7.2f\n", "FADD TPUT ILP=4", ft4, (double)ft4 / NMMA);
  printf("  %-22s  wall=%10lld   cyc/fadd=%7.2f\n", "FADD TPUT ILP=8", ft8, (double)ft8 / NMMA);
  printf("  %-22s  wall=%10lld   cyc/fadd=%7.2f\n", "FADD TPUT ILP=16", ft16, (double)ft16 / NMMA);

  double L = (double)lat / NMMA;
  double T = (double)tp8 / NMMA;
  if (tp16 > 0) T = std::min(T, (double)tp16 / NMMA);
  T = std::min(T, (double)tp4 / NMMA);
  if (T < 1e-6) T = (double)tp8 / NMMA;
  double Lf = (double)fl / NMMA;
  double Tf = std::min((double)ft16 / NMMA, (double)ft8 / NMMA);
  printf("\n  measured:  L_tc=%.2f  T_tc=%.2f  L/T=%.2f  (need >=%.0f independent mma)\n", L, T,
         L / T, ceil(L / T));
  printf("             L_fadd=%.2f  T_fadd=%.2f\n", Lf, Tf);

  printf("\n== 2. MMA + epilogue (acc += d0) ==\n");
  printf("  -- 1 chain serial --\n");
  long long e1 = bench1(kEpi1, dA, dB, dO, dC);
  pr("EPI serial", e1, NMMA);

  printf("  -- interleave (mma;fadd;mma;fadd) independent regs --\n");
  long long i2 = bench1(kEpiInterleave<2>, dA, dB, dO, dC);
  long long i4 = bench1(kEpiInterleave<4>, dA, dB, dO, dC);
  long long i8 = bench1(kEpiInterleave<8>, dA, dB, dO, dC);
  pr("EPI interleave2", i2, NMMA);
  pr("EPI interleave4  [=naive]", i4, NMMA);
  pr("EPI interleave8", i8, NMMA);

  printf("  -- grouped (mma x N; fadd x N)  software ILP --\n");
  long long g2 = bench1(kEpiGrouped<2>, dA, dB, dO, dC);
  long long g4 = bench1(kEpiGrouped<4>, dA, dB, dO, dC);
  long long g8 = bench1(kEpiGrouped<8>, dA, dB, dO, dC);
  long long g16 = bench1(kEpiGrouped<16>, dA, dB, dO, dC);
  pr("EPI grouped2", g2, NMMA);
  pr("EPI grouped4   [=ILP]", g4, NMMA);
  pr("EPI grouped8", g8, NMMA);
  pr("EPI grouped16", g16, NMMA);

  printf("  -- TLP same SMSP (hardware switches warps) --\n");
  long long ts[4];
  int wi = 0;
  for (int w : {1, 2, 4, 8}) {
    ts[wi] = bench_tlp(kEpiTLP_same, dA, dB, dO, dC, w);
    char buf[64];
    snprintf(buf, sizeof(buf), "EPI TLP=%d same", w);
    pr(buf, ts[wi], NMMA);
    wi++;
  }

  printf("  -- TLP spread SMSP (uses more TC units; NOT a fair ILP match) --\n");
  for (int w : {1, 2, 4}) {
    long long t = bench_tlp(kEpiTLP_spread, dA, dB, dO, dC, w);
    char buf[64];
    snprintf(buf, sizeof(buf), "EPI TLP=%d spread", w);
    pr(buf, t, NMMA);
  }

  double c_serial = (double)e1 / NMMA;
  double c_ilv4 = (double)i4 / NMMA;
  double c_grp4 = (double)g4 / NMMA;
  double c_tlp4 = (double)ts[2] / NMMA;
  // 1 链：下一条 mma 和 fadd 抢同一个 d0（WAR），关键路径 ≈ L_tc + L_fadd
  double pred_serial = L + Lf;
  // 独立 N 链：ptxas 会重排 mma/fadd，source 是 interleave 还是 grouped 生成同一套 SASS。
  // 稳态 = TC issue 间隔；N >= L/T ≈ 2 就够。fadd 走 CUDA core，只多占 issue 槽，
  // 每 mma 大约 +T_fadd，N 变大后摊薄。
  auto pred_ilp = [&](int n) { return std::max(T, L / n); };
  printf("\n== 3. theory vs measured (cyc/mma) ==\n");
  printf("  serial         pred=L_tc+L_fadd=%.2f           meas=%.2f\n", pred_serial, c_serial);
  printf("  interleave4    pred=max(T,L/4)=%.2f  (ptxas 重排后==grouped) meas=%.2f\n", pred_ilp(4),
         c_ilv4);
  printf("  grouped2       pred=max(T,L/2)=%.2f            meas=%.2f\n", pred_ilp(2),
         (double)g2 / NMMA);
  printf("  grouped4       pred=max(T,L/4)=%.2f            meas=%.2f\n", pred_ilp(4), c_grp4);
  printf("  grouped8       pred=max(T,L/8)=%.2f            meas=%.2f\n", pred_ilp(8),
         (double)g8 / NMMA);
  printf("  grouped16      pred=max(T,L/16)=%.2f           meas=%.2f\n", pred_ilp(16),
         (double)g16 / NMMA);
  printf("  TLP4 same      pred=max(T,L/4)=%.2f            meas=%.2f\n", pred_ilp(4), c_tlp4);
  printf("  TLP8 same      pred=max(T,L/8)=%.2f            meas=%.2f\n", pred_ilp(8),
         (double)ts[3] / NMMA);
  printf("\n  grouped4 / TLP4 = %.3f   (1.00 = software ILP matches TLP on one SMSP)\n",
         c_grp4 / c_tlp4);
  printf("  grouped8 / TLP8 = %.3f\n", ((double)g8 / NMMA) / ((double)ts[3] / NMMA));
  printf("  grouped16/ T_tc = %.3f\n", ((double)g16 / NMMA) / T);
  return 0;
}
