#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

// Canonical kernel for ILP vs same-SMSP TLP. Full CUDA→PTX→SASS:
//   micro_bench/ilp_vs_tlp_sass.md
//
// kTLP  : <<<1,512>>>, warps 0/4/8/12 only. One MMA chain per warp.
// kHand : <<<1,32>>>,  one warp, four MMA chains (x/y/z/w) in one PTX loop.
//         Unique B (bx=b0+0 .. bw=b0+3) stops ptxas CSE of the four mma.
//         Source order is SWP: clamp(i_{t-1}) → fma(acc_t) → mma → f2i(i_t).
//         ptxas still aliases epi temps (WAW on R25); patch_sass.py splits
//         those dests and spaces the four HMMA by ~27 ops.
//
// MMA : mma.sync.m16n8k16 f32.f16.f16.f32  →  SASS HMMA.16816.F32
// epi : v = relu(d*1.001+0.07); q = clamp(round(v*127), -128, 127)
// time: clock64, cyc/mma = cycles / (4 * rounds)
//
//   nvcc -O3 -arch=sm_120 -o /tmp/swp_hand micro_bench/swp_hand.cu -lcuda
//   nvcc -O3 -arch=sm_120 -cubin -o /tmp/swp_hand.cubin micro_bench/swp_hand.cu
//   python3 micro_bench/patch_sass.py /tmp/swp_hand.cubin kHand -o /tmp/kHand.patched.cubin
//   /tmp/swp_hand /tmp/kHand.patched.cubin

#ifndef NREP
#define NREP 7
#endif

__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3, unsigned a0,
                                    unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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

// One chain's epilogue. kTLP runs this after each MMA.
// Maps 1:1 onto TLP SASS: FFMA, FMNMX(relu), FMUL*127, F2I, VIMNMX clamp, IADD3.
__device__ __forceinline__ int epi_one(float d0, float d1, float d2, float d3) {
  float s = 1.001f, b = 0.07f;
  float v0 = fmaf(d0, s, b), v1 = fmaf(d1, s, b), v2 = fmaf(d2, s, b), v3 = fmaf(d3, s, b);
  v0 = fmaxf(v0, 0.f);
  v1 = fmaxf(v1, 0.f);
  v2 = fmaxf(v2, 0.f);
  v3 = fmaxf(v3, 0.f);
  auto q = [](float v) {
    int i = __float2int_rn(v * 127.f);
    return max(-128, min(127, i));
  };
  return q(v0) + q(v1) + q(v2) + q(v3);
}

extern "C" __global__ void kTLP(unsigned* A, unsigned* B, float* O, long long* C, int rounds) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((warp & 3) != 0) return;   // same SMSP: warps 0,4,8,12
  if ((warp >> 2) >= 4) return;
  unsigned a0 = A[lane * 4], a1 = A[lane * 4 + 1], a2 = A[lane * 4 + 2], a3 = A[lane * 4 + 3];
  unsigned b0 = B[lane * 2], b1 = B[lane * 2 + 1];
  float d0 = 0.01f, d1 = 0, d2 = 0, d3 = 0;
  int sink = 0;
  __syncthreads();
  long long t0 = clk();
#pragma unroll 1
  for (int i = 0; i < rounds; i++) {
    mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    sink += epi_one(d0, d1, d2, d3);
  }
  long long t1 = clk_dep((float)sink);
  if (lane == 0) {
    C[warp] = t1 - t0;
    O[warp] = (float)sink;
  }
}

// One chain (×4 for x/y/z/w). Expanded PTX is in ilp_vs_tlp_sass.md §4.
//   clamp ix* (from last F2I) → add sink
//   fma/relu/×127 acc → t*
//   mma acc
//   cvt t* → ix*   (next iter clamp)
#define PTX_CLAMP(I0, I1, I2, I3, SNK)                                                             \
  "  min.s32 " I0 ", " I0 ", 127;\n"                                                               \
  "  max.s32 " I0 ", " I0 ", -128;\n"                                                              \
  "  min.s32 " I1 ", " I1 ", 127;\n"                                                               \
  "  max.s32 " I1 ", " I1 ", -128;\n"                                                              \
  "  min.s32 " I2 ", " I2 ", 127;\n"                                                               \
  "  max.s32 " I2 ", " I2 ", -128;\n"                                                              \
  "  min.s32 " I3 ", " I3 ", 127;\n"                                                               \
  "  max.s32 " I3 ", " I3 ", -128;\n"                                                              \
  "  add.s32 " SNK ", " SNK ", " I0 ";\n"                                                          \
  "  add.s32 " SNK ", " SNK ", " I1 ";\n"                                                          \
  "  add.s32 " SNK ", " SNK ", " I2 ";\n"                                                          \
  "  add.s32 " SNK ", " SNK ", " I3 ";\n"

#define PTX_FMA(A0, A1, A2, A3, V0, V1, V2, V3, T0, T1, T2, T3)                                    \
  "  fma.rn.f32 " V0 ", " A0 ", 0f3f8020c5, 0f3d8f5c29;\n"                                         \
  "  fma.rn.f32 " V1 ", " A1 ", 0f3f8020c5, 0f3d8f5c29;\n"                                         \
  "  fma.rn.f32 " V2 ", " A2 ", 0f3f8020c5, 0f3d8f5c29;\n"                                         \
  "  fma.rn.f32 " V3 ", " A3 ", 0f3f8020c5, 0f3d8f5c29;\n"                                         \
  "  max.f32 " V0 ", " V0 ", 0f00000000;\n"                                                        \
  "  max.f32 " V1 ", " V1 ", 0f00000000;\n"                                                        \
  "  max.f32 " V2 ", " V2 ", 0f00000000;\n"                                                        \
  "  max.f32 " V3 ", " V3 ", 0f00000000;\n"                                                        \
  "  mul.f32 " T0 ", " V0 ", 0f42fe0000;\n"                                                        \
  "  mul.f32 " T1 ", " V1 ", 0f42fe0000;\n"                                                        \
  "  mul.f32 " T2 ", " V2 ", 0f42fe0000;\n"                                                        \
  "  mul.f32 " T3 ", " V3 ", 0f42fe0000;\n"

#define PTX_F2I(T0, T1, T2, T3, I0, I1, I2, I3)                                                    \
  "  cvt.rni.s32.f32 " I0 ", " T0 ";\n"                                                            \
  "  cvt.rni.s32.f32 " I1 ", " T1 ";\n"                                                            \
  "  cvt.rni.s32.f32 " I2 ", " T2 ";\n"                                                            \
  "  cvt.rni.s32.f32 " I3 ", " T3 ";\n"

#define PTX_MMA(A0, A1, A2, A3, B)                                                                 \
  "  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {" A0 "," A1 "," A2 "," A3                  \
  "},{a0r,a1r,a2r,a3r},{" B ",b1r},{" A0 "," A1 "," A2 "," A3 "};\n"

#define PTX_ROUND()                                                                                \
  PTX_CLAMP("ix0", "ix1", "ix2", "ix3", "sx")                                                      \
  PTX_CLAMP("iy0", "iy1", "iy2", "iy3", "sy")                                                      \
  PTX_CLAMP("iz0", "iz1", "iz2", "iz3", "sz")                                                      \
  PTX_CLAMP("iw0", "iw1", "iw2", "iw3", "sw")                                                      \
  PTX_FMA("x0", "x1", "x2", "x3", "vx0", "vx1", "vx2", "vx3", "tx0", "tx1", "tx2", "tx3")           \
  PTX_FMA("y0", "y1", "y2", "y3", "vy0", "vy1", "vy2", "vy3", "ty0", "ty1", "ty2", "ty3")           \
  PTX_FMA("z0", "z1", "z2", "z3", "vz0", "vz1", "vz2", "vz3", "tz0", "tz1", "tz2", "tz3")           \
  PTX_FMA("w0", "w1", "w2", "w3", "vw0", "vw1", "vw2", "vw3", "tw0", "tw1", "tw2", "tw3")           \
  PTX_MMA("x0", "x1", "x2", "x3", "bx")                                                            \
  PTX_MMA("y0", "y1", "y2", "y3", "by")                                                            \
  PTX_MMA("z0", "z1", "z2", "z3", "bz")                                                            \
  PTX_MMA("w0", "w1", "w2", "w3", "bw")                                                            \
  PTX_F2I("tx0", "tx1", "tx2", "tx3", "ix0", "ix1", "ix2", "ix3")                                   \
  PTX_F2I("ty0", "ty1", "ty2", "ty3", "iy0", "iy1", "iy2", "iy3")                                   \
  PTX_F2I("tz0", "tz1", "tz2", "tz3", "iz0", "iz1", "iz2", "iz3")                                   \
  PTX_F2I("tw0", "tw1", "tw2", "tw3", "iw0", "iw1", "iw2", "iw3")

extern "C" __global__ __launch_bounds__(32) void kHand(unsigned* A, unsigned* B, float* O,
                                                       long long* C, int rounds) {
  int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32) return;
  unsigned a0 = A[lane * 4], a1 = A[lane * 4 + 1], a2 = A[lane * 4 + 2], a3 = A[lane * 4 + 3];
  unsigned b0 = B[lane * 2], b1 = B[lane * 2 + 1];
  int sink = 0;
  long long t0 = 0, t1 = 0;

  asm volatile(
      "{\n"
      "  .reg .f32 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;\n"
      "  .reg .f32 vx0, vx1, vx2, vx3, vy0, vy1, vy2, vy3;\n"
      "  .reg .f32 vz0, vz1, vz2, vz3, vw0, vw1, vw2, vw3;\n"
      "  .reg .f32 tx0, tx1, tx2, tx3, ty0, ty1, ty2, ty3;\n"
      "  .reg .f32 tz0, tz1, tz2, tz3, tw0, tw1, tw2, tw3;\n"
      "  .reg .s32 ix0, ix1, ix2, ix3, iy0, iy1, iy2, iy3;\n"
      "  .reg .s32 iz0, iz1, iz2, iz3, iw0, iw1, iw2, iw3;\n"
      "  .reg .s32 sx, sy, sz, sw, sinkr, rr, nn;\n"
      "  .reg .b32 a0r, a1r, a2r, a3r, b0r, b1r, bx, by, bz, bw;\n"
      "  .reg .pred p;\n"
      "  .reg .u64 c0, c1;\n"
      "  mov.b32 a0r, %3;\n"
      "  mov.b32 a1r, %4;\n"
      "  mov.b32 a2r, %5;\n"
      "  mov.b32 a3r, %6;\n"
      "  mov.b32 b0r, %7;\n"
      "  mov.b32 b1r, %8;\n"
      "  mov.s32 nn, %9;\n"
      "  mov.f32 x0, 0f3c23d70a;\n"
      "  mov.f32 x1, 0f00000000;\n"
      "  mov.f32 x2, 0f00000000;\n"
      "  mov.f32 x3, 0f00000000;\n"
      "  mov.f32 y0, 0f3ca3d70a;\n"
      "  mov.f32 y1, 0f00000000;\n"
      "  mov.f32 y2, 0f00000000;\n"
      "  mov.f32 y3, 0f00000000;\n"
      "  mov.f32 z0, 0f3cf5c28f;\n"
      "  mov.f32 z1, 0f00000000;\n"
      "  mov.f32 z2, 0f00000000;\n"
      "  mov.f32 z3, 0f00000000;\n"
      "  mov.f32 w0, 0f3d23d70a;\n"
      "  mov.f32 w1, 0f00000000;\n"
      "  mov.f32 w2, 0f00000000;\n"
      "  mov.f32 w3, 0f00000000;\n"
      "  add.u32 bx, b0r, 0;\n"
      "  add.u32 by, b0r, 1;\n"
      "  add.u32 bz, b0r, 2;\n"
      "  add.u32 bw, b0r, 3;\n"
      "  mov.s32 sx, 0;\n"
      "  mov.s32 sy, 0;\n"
      "  mov.s32 sz, 0;\n"
      "  mov.s32 sw, 0;\n"
      "  mov.s32 ix0, 0;\n"
      "  mov.s32 ix1, 0;\n"
      "  mov.s32 ix2, 0;\n"
      "  mov.s32 ix3, 0;\n"
      "  mov.s32 iy0, 0;\n"
      "  mov.s32 iy1, 0;\n"
      "  mov.s32 iy2, 0;\n"
      "  mov.s32 iy3, 0;\n"
      "  mov.s32 iz0, 0;\n"
      "  mov.s32 iz1, 0;\n"
      "  mov.s32 iz2, 0;\n"
      "  mov.s32 iz3, 0;\n"
      "  mov.s32 iw0, 0;\n"
      "  mov.s32 iw1, 0;\n"
      "  mov.s32 iw2, 0;\n"
      "  mov.s32 iw3, 0;\n"
      "  mov.s32 rr, 0;\n"
      "  mov.u64 c0, %%clock64;\n"
      "loop:\n"
      PTX_ROUND()
      "  add.s32 rr, rr, 1;\n"
      "  setp.lt.s32 p, rr, nn;\n"
      "  @p bra loop;\n"
      "  add.s32 sinkr, sx, sy;\n"
      "  add.s32 sinkr, sinkr, sz;\n"
      "  add.s32 sinkr, sinkr, sw;\n"
      "  mov.u64 c1, %%clock64;\n"
      "  mov.s32 %0, sinkr;\n"
      "  mov.u64 %1, c0;\n"
      "  mov.u64 %2, c1;\n"
      "}\n"
      : "=r"(sink), "=l"(t0), "=l"(t1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(rounds));

  if (lane == 0) {
    C[0] = t1 - t0;
    O[0] = (float)sink;
  }
}

static long long med(long long* v, int n) {
  std::sort(v, v + n);
  return v[n / 2];
}

static void ck(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = "?";
  cuGetErrorString(r, &s);
  fprintf(stderr, "CUDA driver error %s: %s\n", what, s);
  abort();
}

static long long run_runtime(void (*k)(unsigned*, unsigned*, float*, long long*, int), unsigned* A,
                             unsigned* B, float* O, long long* C, int rounds, int threads) {
  long long h[NREP];
  k<<<1, threads>>>(A, B, O, C, rounds);
  cudaDeviceSynchronize();
  for (int r = 0; r < NREP; r++) {
    cudaMemset(C, 0, 256);
    k<<<1, threads>>>(A, B, O, C, rounds);
    cudaDeviceSynchronize();
    if (threads == 32) {
      cudaMemcpy(&h[r], C, 8, cudaMemcpyDeviceToHost);
    } else {
      long long tmp[16] = {0};
      cudaMemcpy(tmp, C, 128, cudaMemcpyDeviceToHost);
      long long mx = 0;
      for (int i = 0; i < 16; i++)
        if (tmp[i] > mx) mx = tmp[i];
      h[r] = mx;
    }
  }
  return med(h, NREP);
}

static long long run_mod(CUfunction f, CUdeviceptr A, CUdeviceptr B, CUdeviceptr O, CUdeviceptr C,
                         int rounds, int threads) {
  long long h[NREP];
  void* args[] = {&A, &B, &O, &C, &rounds};
  ck(cuLaunchKernel(f, 1, 1, 1, threads, 1, 1, 0, 0, args, 0), "launch warmup");
  ck(cuCtxSynchronize(), "sync warmup");
  for (int r = 0; r < NREP; r++) {
    ck(cuMemsetD8(C, 0, 256), "memset");
    ck(cuLaunchKernel(f, 1, 1, 1, threads, 1, 1, 0, 0, args, 0), "launch");
    ck(cuCtxSynchronize(), "sync");
    if (threads == 32) {
      ck(cuMemcpyDtoH(&h[r], C, 8), "d2h");
    } else {
      long long tmp[16] = {0};
      ck(cuMemcpyDtoH(tmp, C, 128), "d2h");
      long long mx = 0;
      for (int i = 0; i < 16; i++)
        if (tmp[i] > mx) mx = tmp[i];
      h[r] = mx;
    }
  }
  return med(h, NREP);
}

int main(int argc, char** argv) {
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

  CUfunction fPatch = nullptr;
  if (argc > 1) {
    ck(cuInit(0), "cuInit");
    CUcontext ctx;
    ck(cuCtxGetCurrent(&ctx), "ctx");
    CUmodule mod;
    ck(cuModuleLoad(&mod, argv[1]), "load cubin");
    ck(cuModuleGetFunction(&fPatch, mod, "kHand"), "get kHand");
  }

  printf("%-8s %10s %10s %10s  ptx/tlp  sass/tlp\n", "rounds", "HandPTX", "TLP", "HandSASS");
  for (int R : {4, 8, 32, 256, 4096}) {
    long long ptx = run_runtime(kHand, dA, dB, dO, dC, R, 32);
    long long tlp = run_runtime(kTLP, dA, dB, dO, dC, R, 512);
    long long sass = 0;
    if (fPatch) {
      sass = run_mod(fPatch, (CUdeviceptr)dA, (CUdeviceptr)dB, (CUdeviceptr)dO, (CUdeviceptr)dC, R,
                     32);
    }
    int n = 4 * R;
    printf("R=%-6d %10.2f %10.2f %10.2f  %6.3f   %6.3f\n", R, (double)ptx / n, (double)tlp / n,
           fPatch ? (double)sass / n : 0.0, (double)ptx / tlp,
           fPatch ? (double)sass / tlp : 0.0);
  }
  return 0;
}
