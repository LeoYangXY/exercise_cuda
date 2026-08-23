# NVLink P2P store 微基准 —— H 判决实验（C1–C4，总字节相同）

机器：2×H20 (sm_90)，单进程，`cudaDeviceEnablePeerAccess(0->1)`，GPU0 对 GPU1 指针发 `st.global`。
**四种 launch 写的总字节数相同**（同一块 256MB dst，REPS=4 遍），所以 1 warp 发的指令最多、C4 最少。

## 1. 拓扑 / NVLink（实测）
```
$ nvidia-smi topo -m   (GPU0/GPU1 行)
        GPU0  GPU1  ...
GPU0     X    NV18  ...
GPU1    NV18    X   ...
=> GPU0<->GPU1 = NV18，18 条 NVLink。

$ nvidia-smi nvlink -s   (GPU0，每条)
   Link 0..17: 26.562 GB/s   (共 18 条)
=> NVLink 单方向峰值参考 = 18 × 26.562 = 478.1 GB/s（~450 常用值）
```
`SMS=78`，`canAccessPeer 0->1=1`，peer 启用成功；远程 store 校验 mismatches=0（真写到 GPU1）。

## 2. 编译与运行
```bash
cd learn-cuda-cute-triton/micro_bench
nvcc -O3 -arch=sm_90 -o m_p2pstore2 m_p2pstore2.cu
./m_p2pstore2
```
- `n = 2^24 = 16,777,216` 个 int4（256MB），是 1024 的倍数 → U 任意都能整除覆盖。
- 覆盖方式：把 `[0,n)` 切成 `W=32*U` int4 的「warp-unit」，每个活跃 warp 认领 `unit ≡ warp_global (mod Wp)` 的 unit，
  每个 unit 恰好写一次 → **四种配置总字节严格相等**（n×REPS×sizeof）。
- C2 用 16-warp block 但只留 warp 0/4/8/12（同 SMSP），闲置 warp 立即 return，不进 stride、不写 dst。
- C3 用 128-thread block（warp 0/1/2/3 各占一个 subcore）。C2/C3 的 A 同=128、每 warp 指令数同，唯一差别是 1 个 vs 4 个 subcore。
- store-only：寄存器值直接 `st`（无 `ld`，H 的最利情形）；copy：`ld src + st dst`。

## 3. 完整源码（`m_p2pstore2.cu`）
```cpp
// Build: nvcc -O3 -arch=sm_90 -o m_p2pstore2 m_p2pstore2.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

typedef int4 vec_t;
static const size_t PAYLOAD   = 1UL << 28;
static const size_t N_INT4    = PAYLOAD / sizeof(vec_t); // 2^24
static const int    REPS      = 4;
static const int    ITERS     = 3;
static const double NVL_PEAK  = 18.0 * 26.562;

#define CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } }while(0)

__global__ void kInit(vec_t* a, size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=s) a[i]=make_int4((int)i,(int)i,(int)i,(int)i);
}
__global__ void kCheck(const vec_t* dst, size_t n, int* mm){
  int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=16) return;
  size_t idx=(size_t)k*(n/16);
  vec_t v=dst[idx], e=make_int4((int)idx,(int)idx,(int)idx,(int)idx);
  if(v.x!=e.x||v.y!=e.y||v.z!=e.z||v.w!=e.w) atomicAdd(mm,1);
}
template <bool PIN, int U, bool STORE_ONLY>
__global__ void kStore(vec_t* __restrict__ dst, const vec_t* __restrict__ src,
                       size_t n, int block_active_warps){
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if(PIN && (warp&3)!=0) return;
  int awarp = PIN ? (warp>>2) : warp;
  int warp_global = blockIdx.x * block_active_warps + awarp;
  size_t Wp = (size_t)gridDim.x * block_active_warps;
  size_t W  = 32ULL * U;
  size_t nunits = n / W;
  size_t unit = (size_t)warp_global;
  for(int rep=0; rep<REPS; ++rep){
    for(; unit<nunits; unit+=Wp){
      size_t B = unit * W;
      #pragma unroll
      for(int u=0; u<U; ++u){
        size_t idx = B + (size_t)u*32 + lane;
        vec_t v;
        if(STORE_ONLY) v = make_int4((int)idx,(int)idx,(int)idx,(int)idx);
        else           v = src[idx];
        dst[idx] = v;
        asm volatile("" ::: "memory");
      }
    }
    unit = (size_t)warp_global;
  }
}
template <bool PIN, bool STORE_ONLY>
double runU(int U, int blocks, int block_threads, int block_active_warps, vec_t* dst, const vec_t* src){
  cudaEvent_t e0,e1; CHK(cudaEventCreate(&e0)); CHK(cudaEventCreate(&e1));
  auto call=[&](int u){
    if(u==1)      kStore<PIN,1 ,STORE_ONLY><<<blocks,block_threads>>>(dst,src,N_INT4,block_active_warps);
    else if(u==4) kStore<PIN,4 ,STORE_ONLY><<<blocks,block_threads>>>(dst,src,N_INT4,block_active_warps);
    else if(u==8) kStore<PIN,8 ,STORE_ONLY><<<blocks,block_threads>>>(dst,src,N_INT4,block_active_warps);
    else if(u==16)kStore<PIN,16,STORE_ONLY><<<blocks,block_threads>>>(dst,src,N_INT4,block_active_warps);
    else          kStore<PIN,32,STORE_ONLY><<<blocks,block_threads>>>(dst,src,N_INT4,block_active_warps);
  };
  call(U); CHK(cudaDeviceSynchronize());
  CHK(cudaEventRecord(e0));
  for(int it=0; it<ITERS; ++it) call(U);
  CHK(cudaEventRecord(e1)); CHK(cudaEventSynchronize(e1));
  float ms; CHK(cudaEventElapsedTime(&ms,e0,e1));
  CHK(cudaEventDestroy(e0)); CHK(cudaEventDestroy(e1));
  double bytes=(double)N_INT4*sizeof(vec_t)*REPS*ITERS;
  return bytes/1e9/(ms/1000.0);
}
static void row(const char*cfg,const char*pay,const char*dstn,int sms,int nw,int A,int U,double g,bool nvl,const char*notes){
  printf("%-5s %-7s %-4s %4d %4d %6d %3d %8.1f %8.1f%%  %7.4f  %s\n",
         cfg,pay,dstn,sms,nw,A,U,g, nvl?100.0*g/NVL_PEAK:0.0, 32.0/(double)A, notes);
}
int main(){
  int devs; CHK(cudaGetDeviceCount(&devs)); if(devs<2){printf("need >=2 GPUs\n");return 1;}
  cudaDeviceProp p; CHK(cudaGetDeviceProperties(&p,0));
  int sms=p.multiProcessorCount;
  int can01=0; CHK(cudaDeviceCanAccessPeer(&can01,0,1));
  printf("=== env ===\nGPU0=%s cap=%d.%d  SMS=%d  canAccessPeer0->1=%d  NVpeak=%.1f GB/s (18x26.562)\n",
         p.name,p.major,p.minor,sms,can01,NVL_PEAK);
  CHK(cudaSetDevice(0));
  if(cudaDeviceEnablePeerAccess(1,0)!=cudaSuccess){ printf("peer access FAILED; stop.\n"); return 1; }
  printf("peer access 0->1 OK\n");
  vec_t *d_src,*d_peer,*d_loc;
  CHK(cudaSetDevice(0)); CHK(cudaMalloc(&d_src,PAYLOAD)); CHK(cudaMalloc(&d_loc,PAYLOAD));
  CHK(cudaSetDevice(1)); CHK(cudaMalloc(&d_peer,PAYLOAD));
  CHK(cudaSetDevice(0)); kInit<<<(N_INT4+255)/256,256>>>(d_src,N_INT4); CHK(cudaDeviceSynchronize());
  runU<false,true>(8,1,128,4,d_peer,d_src); CHK(cudaDeviceSynchronize());
  int* d_mm; CHK(cudaMalloc(&d_mm,4)); CHK(cudaMemset(d_mm,0,4));
  kCheck<<<1,32>>>(d_peer,N_INT4,d_mm); CHK(cudaDeviceSynchronize());
  int h_mm=0; CHK(cudaMemcpy(&h_mm,d_mm,4,cudaMemcpyDeviceToHost));
  printf("correctness remote store-only: mismatches=%d %s\n\n",h_mm,h_mm==0?"OK":"FAIL");
  if(h_mm!=0){printf("remote write failed; stop.\n");return 1;}
  printf("%-5s %-7s %-4s %4s %4s %6s %3s %8s %9s  %7s  %s\n",
         "cfg","pay","dst","sms","nw","A","U","GB/s","%nvl","inst/w","notes");
  printf("-----------------------------------------------------------------------------------------------\n");
  int Ulist[]={1,4,8,16,32};
  for(int U:Ulist){ row("C1","so","nvl",1,1,32,U, runU<false,true>(U,1,32,1,d_peer,d_src), true,"1w store-only"); }
  for(int U:{1,8,32}){ row("C1","cp","nvl",1,1,32,U, runU<false,false>(U,1,32,1,d_peer,d_src), true,"1w copy"); }
  for(int U:Ulist){ row("C2","so","nvl",1,4,128,U, runU<true ,true>(U,1,512,4,d_peer,d_src), true,"4w 1subcore so"); }
  for(int U:{1,8,32}){ row("C2","cp","nvl",1,4,128,U, runU<true ,false>(U,1,512,4,d_peer,d_src), true,"4w 1subcore cp"); }
  for(int U:Ulist){ row("C3","so","nvl",1,4,128,U, runU<false,true>(U,1,128,4,d_peer,d_src), true,"4subcore so"); }
  for(int U:{1,8,32}){ row("C3","cp","nvl",1,4,128,U, runU<false,false>(U,1,128,4,d_peer,d_src), true,"4subcore cp"); }
  int smlist[]={2,4,8,16,32,78};
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,4*b,128*b,8, runU<false,true>(8,b,128,4,d_peer,d_src), true,"4w/blk U8"); if(b==sms)break; }
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,32*b,1024*b,8, runU<false,true>(8,b,1024,32,d_peer,d_src), true,"32w/blk U8"); if(b==sms)break; }
  for(int s:{2,8,78}){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,4*b,128*b,1, runU<false,true>(1,b,128,4,d_peer,d_src), true,"4w/blk U1"); }
  row("C1","so","loc",1,1,32,32, runU<false,true>(32,1,32,1,d_loc,d_src), false,"local 1w U32");
  row("C2","so","loc",1,4,128,32, runU<true ,true>(32,1,512,4,d_loc,d_src), false,"local 4w1sub U32");
  row("C3","so","loc",1,4,128,32, runU<false,true>(32,1,128,4,d_loc,d_src), false,"local 4sub U32");
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","loc",b,4*b,128*b,8, runU<false,true>(8,b,128,4,d_loc,d_src), false,"local 4w/blk U8"); if(b==sms)break; }
  printf("\nNVpeak=%.1f GB/s. %%nvl only meaningful for remote. inst/w = 32/A (per-warp iterations vs C1).\n",NVL_PEAK);
  CHK(cudaSetDevice(0)); CHK(cudaFree(d_src)); CHK(cudaFree(d_loc));
  CHK(cudaSetDevice(1)); CHK(cudaFree(d_peer));
  printf("done.\n"); return 0;
}
```

## 4. 原始表格（实测）
```
cfg   pay     dst   sms   nw      A   U     GB/s      %nvl   inst/w  notes
-----------------------------------------------------------------------------------------------
C1    so      nvl     1    1     32   1     20.7      4.3%   1.0000  1w store-only
C1    so      nvl     1    1     32   4     50.3     10.5%   1.0000  1w store-only
C1    so      nvl     1    1     32   8     50.6     10.6%   1.0000  1w store-only
C1    so      nvl     1    1     32  16     50.6     10.6%   1.0000  1w store-only
C1    so      nvl     1    1     32  32     50.6     10.6%   1.0000  1w store-only
C1    cp      nvl     1    1     32   1      1.3      0.3%   1.0000  1w copy
C1    cp      nvl     1    1     32   8      4.8      1.0%   1.0000  1w copy
C1    cp      nvl     1    1     32  32      4.8      1.0%   1.0000  1w copy
C2    so      nvl     1    4    128   1     50.6     10.6%   0.2500  4w 1subcore so
C2    so      nvl     1    4    128   4     50.6     10.6%   0.2500  4w 1subcore so
C2    so      nvl     1    4    128   8     50.6     10.6%   0.2500  4w 1subcore so
C2    so      nvl     1    4    128  16     50.6     10.6%   0.2500  4w 1subcore so
C2    so      nvl     1    4    128  32     50.6     10.6%   0.2500  4w 1subcore so
C2    cp      nvl     1    4    128   1      5.0      1.1%   0.2500  4w 1subcore cp
C2    cp      nvl     1    4    128   8     18.3      3.8%   0.2500  4w 1subcore cp
C2    cp      nvl     1    4    128  32     18.4      3.9%   0.2500  4w 1subcore cp
C3    so      nvl     1    4    128   1     50.6     10.6%   0.2500  4subcore so
C3    so      nvl     1    4    128   4     50.6     10.6%   0.2500  4subcore so
C3    so      nvl     1    4    128   8     50.6     10.6%   0.2500  4subcore so
C3    so      nvl     1    4    128  16     50.6     10.6%   0.2500  4subcore so
C3    so      nvl     1    4    128  32     50.6     10.6%   0.2500  4subcore so
C3    cp      nvl     1    4    128   1      5.0      1.1%   0.2500  4subcore cp
C3    cp      nvl     1    4    128   8     18.3      3.8%   0.2500  4subcore cp
C3    cp      nvl     1    4    128  32     18.5      3.9%   0.2500  4subcore cp
C4    so      nvl     2    8    256   8    101.3     21.2%   0.1250  4w/blk U8
C4    so      nvl     4   16    512   8    202.4     42.3%   0.0625  4w/blk U8
C4    so      nvl     8   32   1024   8    309.5     64.7%   0.0312  4w/blk U8
C4    so      nvl    16   64   2048   8    373.7     78.2%   0.0156  4w/blk U8
C4    so      nvl    32  128   4096   8    373.1     78.0%   0.0078  4w/blk U8
C4    so      nvl    78  312   9984   8    367.3     76.8%   0.0032  4w/blk U8
C4    so      nvl     2   64   2048   8    101.3     21.2%   0.0156  32w/blk U8
C4    so      nvl     4  128   4096   8    202.5     42.4%   0.0078  32w/blk U8
C4    so      nvl     8  256   8192   8    314.4     65.8%   0.0039  32w/blk U8
C4    so      nvl    16  512  16384   8    373.4     78.1%   0.0020  32w/blk U8
C4    so      nvl    32 1024  32768   8    373.0     78.0%   0.0010  32w/blk U8
C4    so      nvl    78 2496  79872   8    364.5     76.2%   0.0004  32w/blk U8
C4    so      nvl     2    8    256   1    101.3     21.2%   0.1250  4w/blk U1
C4    so      nvl     8   32   1024   1    312.4     65.3%   0.0312  4w/blk U1
C4    so      nvl    78  312   9984   1    363.0     75.9%   0.0032  4w/blk U1
C1    so      loc     1    1     32  32     63.3      0.0%   1.0000  local 1w U32
C2    so      loc     1    4    128  32     63.3      0.0%   0.2500  local 4w1sub U32
C3    so      loc     1    4    128  32     63.3      0.0%   0.2500  local 4sub U32
C4    so      loc     2    8    256   8    126.6      0.0%   0.1250  local 4w/blk U8
C4    so      loc     4   16    512   8    253.2      0.0%   0.0625  local 4w/blk U8
C4    so      loc     8   32   1024   8    505.9      0.0%   0.0312  local 4w/blk U8
C4    so      loc    16   64   2048   8   1010.4      0.0%   0.0156  local 4w/blk U8
C4    so      loc    32  128   4096   8   2011.5      0.0%   0.0078  local 4w/blk U8
C4    so      loc    78  312   9984   8   3011.2      0.0%   0.0032  local 4w/blk U8
```

## 5. 结论（H 判决）

**核心数字**：单 SM 任意形态（C1/C2/C3）store-only ≈ **50.6 GB/s**；
多 SM（C4，U=8，4w/blk）在 **16 SM 处饱和 ≈ 373.7 GB/s**（之后 32/78 SM 平台 367–373）。
单 SM 本地 store-only ≈ 63.3 GB/s；本地 C4 在 78 SM 仍线性爬到 3011 GB/s（HBM 未被打满）。

1. **C1 最大 ILP store-only 远程 = 50.6 GB/s，仅为 C4 饱和带宽的 13.5%**（< 80% 阈值）。
   → 按证伪标准，**H 不成立**：1 warp 即使拉满在途（U=32），也只到 NVLink 峰值的 ~13%。
   （注：U=1 时 C1 仅 20.7，U≥4 即 50.6——ILP 确有作用，把 1 warp 从「stall」抬到「单 SM 发射端口打满」，
   但封顶就是单 SM 对 NVLink 的注入上限 ~50.6，不是 NVLink 天花板。）

2. **C2 vs C1（同 1 个 scheduler）**：U=32 时 C2=50.6 == C1=50.6（**0 增益**）。
   说明 C1 在 U=32 已每拍都在 st，同调度器上再加 4× TLP 无济于事。
   但 U=1 时 C2=50.6 >> C1=20.7（2.4×）——低 ILP 时 TLP 才藏住 C1 的 stall。
   → 「只在 C1 仍 stall 时 C2 才更快」成立：高 ILP 下两者都顶到同一 scheduler 上限。

3. **C3 vs C2（同工作量，4 subcore vs 1 subcore）**：U=32 时 C3=50.6 == C2=50.6（**0 增益**）。
   1 个 SM 内 4 个 scheduler 没能给出 4× issue——单 SM 的 store/MIO 总出口被封在 ~50.6 GB/s，
   subcore 数不是瓶颈，**单 SM 总 store 带宽才是**。C3 没比 C1/C2 快。

4. **C4 饱和：~16 SM，~374 GB/s**（78 SM 不掉，平台 367–373）。
   C4 把「按 SM 份数加」的在途/MIO 累加起来，才触到 NVLink 天花板；单 SM 那 ~50.6 再多 warp 也只 1 份。

5. **总字节相同 → 带宽相同？错。H 不成立。**
   带宽 = 字节 / 时间；时间取决于「同时有几根发射槽、能挂多少在途」。
   1 warp 只是把同样的字节拆成更多次循环，但单 SM 的 store 出口与 store-buffer 在途容量被 NVLink credit 限死在 ~50.6 GB/s，
   必须靠多 SM 把在途按份数叠加（~16 SM）才到 374。字节相同不改变「发射槽/在途」这个真正的时间瓶颈。

6. **Little’s law 校验（L≈1μs）**：
   N_est ≈ BW × L = GB/s × 1e9 × 1e-6 = GB/s × 1000 字节。
   - C1 (50.6) → N_est ≈ **50.6 KB**；BDP = 450 GB/s × 1μs = **450 KB**。
     C1 的在途 ≈ 50.6/450 ≈ **11% 的 BDP** → 远挂不住 BDP → 喂不满 NVLink。
   - C4 饱和 (374) → N_est ≈ **374 KB** ≈ **83% 的 BDP** → 在途够，打满。
   单 SM store-buffer 顶多挂 ~50KB，远小于 450KB；要补齐 ~9 份在途实测需 ~16 SM。与实测吻合。

**一句话**：通信和 TC 是同一套 Little 定律，但 NVLink store 的 L/T 大得多——单 SMSP/单 SM 无论 1 warp ILP 还是 4 warp TLP 都被「单 SM→NVLink 注入上限」封顶（~50 GB/s），只有按 SM 份数叠加在途（~16 SM）才触到 NVLink 的 ~374 GB/s。总字节一样不等于带宽一样；H 被证伪。

> 旁证（copy 形态）：C1 copy 只有 4.8 GB/s，C2/C3 copy 到 18.4 GB/s——copy 多了 `ld` 延迟，1 warp 更藏不住，TLP 帮助更大；但 store-only 下 1 warp 的 ILP 已能顶满单 SM 端口，故 C1≈C2≈C3。
> 本地对照：单 SM 本地 store 63.3 GB/s（与远程 50.6 同量级，说明 1 SM 写吞吐是 LSU/端口受限，与目的地无关）；多 SM 本地 78 SM 达 3011 GB/s（HBM，未饱和），远程 16 SM 就封顶 374（NVLink 注入率）。差别在「多 SM 峰值天花板」，不在单 SM。
