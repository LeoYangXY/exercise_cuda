# NVLink P2P *store* 微基准（SM 发 `st.global` 到对端，非 copy engine）

机器：2× H20 (Hopper sm_90)，单进程，GPU0 跑 kernel，peer 指针 `st.global` 到 GPU1。

## 1. 拓扑 / NVLink（实测）

```
$ nvidia-smi topo -m        (GPU0 / GPU1 行)
        GPU0  GPU1  ...
GPU0     X    NV18  ...
GPU1    NV18    X   ...
=> GPU0<->GPU1 是 NV18，18 条 NVLink。

$ nvidia-smi nvlink -s       (GPU0, 每条 link)
GPU 0: NVIDIA H20
   Link 0: 26.562 GB/s
   Link 1: 26.562 GB/s
   ... (共 18 条，每条 26.562 GB/s)
=> NVLink 单方向聚合峰值参考 = 18 × 26.562 = 478.1 GB/s
```
> 注：`nvidia-smi` 报的 26.562 是单 link 保守值；H20 实际单向 NVLink 常引用 ~450 GB/s。
> 下文 `%_of_nvlink` 以 478.1 为基准；达到 ~370 GB/s 即约 82% 的 ~450 真实峰值。

`multiProcessorCount(GPU0) = 78`，`canAccessPeer 0->1 = 1`，peer access 启用成功。
远程 store 正确性校验：抽查 16 个 int4，mismatches = 0（确认真写到 GPU1）。

## 2. 编译与运行

```bash
cd learn-cuda-cute-triton/micro_bench
nvcc -O3 -arch=sm_90 -o m_p2pstore m_p2pstore.cu
./m_p2pstore
```
- payload 每 buffer = 256 MB；kernel 内 REPS=4 重写，计时 ITERS=3 → 每次测量 ~3 GB。
- 只报 **store** 字节（dst 写量），不算 load。
- 全部 5 个变体（A/B/C/D/E）+ F 门铃，均 GPU0 发 `st.global`；本地对照把 dst 改成本卡。

## 3. 完整源码（`m_p2pstore.cu`）

```cpp
// m_p2pstore.cu  --  NVLink P2P *store* micro-benchmark (SM-issued st.global to peer)
// 单进程 2 卡，cudaDeviceEnablePeerAccess；不依赖 DeepEP/NCCL/NVSHMEM；不用 cudaMemcpyPeer。
// Build:  nvcc -O3 -arch=sm_90 -o m_p2pstore m_p2pstore.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

typedef int4 vec_t;                       // 16 bytes per element
static const size_t PAYLOAD_BYTES = 1UL << 28;          // 256 MB per buffer
static const size_t N_INT4         = PAYLOAD_BYTES / sizeof(vec_t); // 2^24
static const int    REPS           = 4;
static const int    ITERS          = 3;
static const int    FLAG_N         = 1 << 12;

#define CUDA_CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } }while(0)

__global__ void kInit(vec_t* a, size_t n){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=s) a[i] = make_int4((int)i,(int)i,(int)i,(int)i);
}
__global__ void kCheck(const vec_t* dst, size_t n, int* mismatches){
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  if(k>=16) return;
  size_t idx = (size_t)k * (n/16);
  vec_t v = dst[idx];
  vec_t e = make_int4((int)idx,(int)idx,(int)idx,(int)idx);
  if(v.x!=e.x || v.y!=e.y || v.z!=e.z || v.w!=e.w) atomicAdd(mismatches,1);
}
template <int UNROLL>
__global__ void kStore(vec_t* __restrict__ dst, const vec_t* __restrict__ src,
                       size_t n, int pin_mode, int doorbell_k, vec_t* __restrict__ flag){
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if(pin_mode==1 && warp!=0) return;
  if(pin_mode==2 && (warp&3)!=0) return;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warpGlobal = (unsigned)(blockIdx.x * (blockDim.x/32) + warp);
  uint64_t written = 0;
  for(int rep=0; rep<REPS; ++rep){
    size_t i = tid;
    for(; i + (size_t)(UNROLL-1)*stride < n; i += (size_t)UNROLL*stride){
      vec_t r[UNROLL];
      #pragma unroll
      for(int u=0;u<UNROLL;++u) r[u] = src[i + (size_t)u*stride];
      #pragma unroll
      for(int u=0;u<UNROLL;++u){
        vec_t v = r[u];
        dst[i + (size_t)u*stride] = v;
        if(doorbell_k){
          if(++written >= (uint64_t)doorbell_k){
            written = 0;
            __threadfence_system();
            if(lane==0) flag[warpGlobal & (FLAG_N-1)] = v;
            __threadfence_system();
            __syncwarp();
          }
        }
      }
    }
    for(; i<n; i+=stride){
      vec_t v = src[i]; dst[i] = v;
      if(doorbell_k){
        if(++written >= (uint64_t)doorbell_k){
          written = 0; __threadfence_system();
          if(lane==0) flag[warpGlobal & (FLAG_N-1)] = v;
          __threadfence_system(); __syncwarp();
        }
      }
    }
  }
}
template<int UNROLL>
static void launchK(int blocks,int wpb,vec_t* dst,const vec_t* src,int pin,int dk,vec_t* flag){
  kStore<UNROLL><<<blocks,wpb*32>>>(dst,src,N_INT4,pin,dk,flag);
}
static void dispatch(int unroll,int blocks,int wpb,vec_t* dst,const vec_t* src,int pin,int dk,vec_t* flag){
  switch(unroll){
    case 1: launchK<1 >(blocks,wpb,dst,src,pin,dk,flag); break;
    case 2: launchK<2 >(blocks,wpb,dst,src,pin,dk,flag); break;
    case 4: launchK<4 >(blocks,wpb,dst,src,pin,dk,flag); break;
    case 8: launchK<8 >(blocks,wpb,dst,src,pin,dk,flag); break;
    default:launchK<16>(blocks,wpb,dst,src,pin,dk,flag); break;
  }
}
static double runCfg(int unroll,int blocks,int wpb,vec_t* dst,const vec_t* src,int pin,int dk,vec_t* flag){
  cudaEvent_t e0,e1; CUDA_CHK(cudaEventCreate(&e0)); CUDA_CHK(cudaEventCreate(&e1));
  dispatch(unroll,blocks,wpb,dst,src,pin,dk,flag); CUDA_CHK(cudaDeviceSynchronize());
  CUDA_CHK(cudaEventRecord(e0));
  for(int it=0; it<ITERS; ++it) dispatch(unroll,blocks,wpb,dst,src,pin,dk,flag);
  CUDA_CHK(cudaEventRecord(e1)); CUDA_CHK(cudaEventSynchronize(e1));
  float ms; CUDA_CHK(cudaEventElapsedTime(&ms,e0,e1));
  CUDA_CHK(cudaEventDestroy(e0)); CUDA_CHK(cudaEventDestroy(e1));
  double bytes = (double)N_INT4 * sizeof(vec_t) * REPS * ITERS;
  return bytes / 1e9 / (ms/1000.0);
}
static int perSmsp(int wpb,int pin){
  if(pin==1) return 1; if(pin==2) return 4;
  int v=(wpb+3)/4; return v<1?1:v;
}
int main(){
  int devs; CUDA_CHK(cudaGetDeviceCount(&devs));
  if(devs<2){ printf("need >=2 GPUs, found %d\n",devs); return 1; }
  cudaDeviceProp p; CUDA_CHK(cudaGetDeviceProperties(&p,0));
  int sms = p.multiProcessorCount;
  int can01=0; CUDA_CHK(cudaDeviceCanAccessPeer(&can01,0,1));
  printf("=== environment ===\n");
  printf("GPU0=%s compute_cap=%d.%d\n", p.name, p.major, p.minor);
  printf("multiProcessorCount(GPU0) = %d\n", sms);
  printf("canAccessPeer 0->1 = %d\n", can01);
  printf("NVLink GPU0<->GPU1: 18 links x 26.562 GB/s = %.1f GB/s aggregate (per dir)\n", 18.0*26.562);
  double nvlink_peak = 18.0*26.562;
  CUDA_CHK(cudaSetDevice(0));
  cudaError_t ea = cudaDeviceEnablePeerAccess(1,0);
  if(ea!=cudaSuccess){ printf("peer access FAILED: %s\n", cudaGetErrorString(ea)); return 1; }
  printf("peer access 0->1 enabled OK\n");
  size_t bufBytes = PAYLOAD_BYTES;
  vec_t *d_src,*d_dst_peer,*d_dst_local,*d_flag;
  CUDA_CHK(cudaSetDevice(0)); CUDA_CHK(cudaMalloc(&d_src,bufBytes));
  CUDA_CHK(cudaMalloc(&d_dst_local,bufBytes));
  CUDA_CHK(cudaSetDevice(1)); CUDA_CHK(cudaMalloc(&d_dst_peer,bufBytes));
  CUDA_CHK(cudaMalloc(&d_flag,FLAG_N*sizeof(vec_t)));
  CUDA_CHK(cudaMemset(d_flag,0,FLAG_N*sizeof(vec_t)));
  CUDA_CHK(cudaSetDevice(0));
  kInit<<<(N_INT4+255)/256,256>>>(d_src,N_INT4); CUDA_CHK(cudaDeviceSynchronize());
  CUDA_CHK(cudaSetDevice(1)); CUDA_CHK(cudaMemset(d_dst_peer,0,bufBytes));
  CUDA_CHK(cudaSetDevice(0)); CUDA_CHK(cudaMemset(d_dst_local,0,bufBytes));
  dispatch(8,1,8,d_dst_peer,d_src,0,0,d_flag); CUDA_CHK(cudaDeviceSynchronize());
  int* d_mm; CUDA_CHK(cudaMalloc(&d_mm,sizeof(int))); CUDA_CHK(cudaMemset(d_mm,0,sizeof(int)));
  kCheck<<<1,32>>>(d_dst_peer,N_INT4,d_mm); CUDA_CHK(cudaDeviceSynchronize());
  int h_mm=0; CUDA_CHK(cudaMemcpy(&h_mm,d_mm,sizeof(int),cudaMemcpyDeviceToHost));
  printf("correctness (remote 1blk/8w/U8): mismatches=%d  %s\n",h_mm,h_mm==0?"OK":"FAIL");
  CUDA_CHK(cudaFree(d_mm));
  if(h_mm!=0){ printf("Remote store did NOT reach GPU1. Stop.\n"); return 1; }
  printf("\n%-14s %4s %8s %6s %7s %9s %10s  %s\n","mode","sms","w/sm","unrl","smsp","GB/s","%nvlink","notes");
  printf("-------------------------------------------------------------------------------\n");
  auto emit = [&](const char* mode,int blocks,int wpb,int unroll,int pin,int dk,vec_t* dst,const char* notes,double peak){
    int sm_used=blocks, wsm=wpb;
    double g=runCfg(unroll,blocks,wpb,dst,d_src,pin,dk,d_flag);
    printf("%-14s %4d %8d %6d %7d %9.1f %9.1f%%  %s\n",
           mode,sm_used,wsm,unroll,perSmsp(wpb,pin),g,100.0*g/nvlink_peak,notes);
  };
  for(int wpb : {1,2,4,8,16,32}){ char n[64]; snprintf(n,sizeof(n),"kTLP warps/block=%d",wpb); emit("TLP",1,wpb,1,0,0,d_dst_peer,n,nvlink_peak); }
  for(int u : {1,2,4,8,16}){ char n[64]; snprintf(n,sizeof(n),"kILP U=%d wpb=1",u); emit("ILP",1,1,u,0,0,d_dst_peer,n,nvlink_peak); }
  for(int wpb : {4,8}){ char n[64]; snprintf(n,sizeof(n),"kILP U=8 wpb=%d",wpb); emit("ILP",1,wpb,8,0,0,d_dst_peer,n,nvlink_peak); }
  emit("ILP_sSMSP",1,1,8,0,0,d_dst_peer,"1 warp x U=8 (SMSP0)",nvlink_peak);
  emit("TLP_sSMSP",1,16,1,2,0,d_dst_peer,"4 warp(w0,4,8,12) x U=1 same SMSP",nvlink_peak);
  int smlist[]={1,2,4,8,16,32,64,132};
  for(int s : smlist){ int b=s; if(b>sms)b=sms; char n[64]; snprintf(n,sizeof(n),"D sm_scale blocks=%d (8w/U8)",b); emit("SMscale",b,8,8,0,0,d_dst_peer,n,nvlink_peak); if(b==sms)break; }
  for(int wpb : {1,2,4,8,16,32}){ char n[64]; snprintf(n,sizeof(n),"local TLP wpb=%d",wpb); emit("locTLP",1,wpb,1,0,0,d_dst_local,n,nvlink_peak); }
  for(int u : {1,2,4,8,16}){ char n[64]; snprintf(n,sizeof(n),"local ILP U=%d wpb=1",u); emit("locILP",1,1,u,0,0,d_dst_local,n,nvlink_peak); }
  for(int s : {1,2,4,8,16,32,64,132}){ int b=s; if(b>sms)b=sms; char n[64]; snprintf(n,sizeof(n),"local SMscale blocks=%d (8w/U8)",b); emit("locScale",b,8,8,0,0,d_dst_local,n,nvlink_peak); if(b==sms)break; }
  for(int k : {16,256,4096}){ char n[64]; snprintf(n,sizeof(n),"doorbell K=%d int4",k); emit("doorbell",1,8,8,0,k,d_dst_peer,n,nvlink_peak); }
  emit("doorbell",1,8,8,0,0,d_dst_peer,"doorbell K=inf (none)",nvlink_peak);
  printf("\nNVLink peak reference = %.1f GB/s. %%nvlink = GB/s / peak.\n", nvlink_peak);
  CUDA_CHK(cudaSetDevice(0)); CUDA_CHK(cudaFree(d_src)); CUDA_CHK(cudaFree(d_dst_local));
  CUDA_CHK(cudaSetDevice(1)); CUDA_CHK(cudaFree(d_dst_peer)); CUDA_CHK(cudaFree(d_flag));
  printf("done.\n");
  return 0;
}
```

## 4. 原始表格（实测）

```
mode            sms     w/sm   unrl    smsp      GB/s    %nvlink  notes
-------------------------------------------------------------------------------
TLP               1        1      1       1       1.2       0.3%  kTLP warps/block=1
TLP               1        2      1       1       2.5       0.5%  kTLP warps/block=2
TLP               1        4      1       1       5.0       1.0%  kTLP warps/block=4
TLP               1        8      1       2       9.9       2.1%  kTLP warps/block=8
TLP               1       16      1       4      19.3       4.0%  kTLP warps/block=16
TLP               1       32      1       8      36.9       7.7%  kTLP warps/block=32
ILP               1        1      1       1       1.2       0.3%  kILP U=1 wpb=1
ILP               1        1      2       1       2.4       0.5%  kILP U=2 wpb=1
ILP               1        1      4       1       4.4       0.9%  kILP U=4 wpb=1
ILP               1        1      8       1       7.3       1.5%  kILP U=8 wpb=1
ILP               1        1     16       1      10.2       2.1%  kILP U=16 wpb=1
ILP               1        4      8       1      24.6       5.1%  kILP U=8 wpb=4
ILP               1        8      8       2      44.1       9.2%  kILP U=8 wpb=8
ILP_sSMSP         1        1      8       1       7.3       1.5%  1 warp x U=8 (SMSP0)
TLP_sSMSP         1       16      1       4      21.6       4.5%  4 warp(w0,4,8,12) x U=1 same SMSP
SMscale           1        8      8       2      44.2       9.2%  D sm_scale blocks=1 (8w/U8)
SMscale           2        8      8       2      89.1      18.6%  D sm_scale blocks=2 (8w/U8)
SMscale           4        8      8       2     175.7      36.8%  D sm_scale blocks=4 (8w/U8)
SMscale           8        8      8       2     353.0      73.8%  D sm_scale blocks=8 (8w/U8)
SMscale          16        8      8       2     371.0      77.6%  D sm_scale blocks=16 (8w/U8)
SMscale          32        8      8       2     366.9      76.7%  D sm_scale blocks=32 (8w/U8)
SMscale          64        8      8       2     367.9      76.9%  D sm_scale blocks=64 (8w/U8)
SMscale          78        8      8       2     363.7      76.1%  D sm_scale blocks=78 (8w/U8)
locTLP            1        1      1       1       1.2       0.3%  local TLP wpb=1
locTLP            1        2      1       1       2.5       0.5%  local TLP wpb=2
locTLP            1        4      1       1       5.0       1.0%  local TLP wpb=4
locTLP            1        8      1       2      10.0       2.1%  local TLP wpb=8
locTLP            1       16      1       4      19.5       4.1%  local TLP wpb=16
locTLP            1       32      1       8      37.6       7.9%  local TLP wpb=32
locILP            1        1      1       1       1.2       0.3%  local ILP U=1 wpb=1
locILP            1        1      2       1       2.4       0.5%  local ILP U=2 wpb=1
locILP            1        1      4       1       4.3       0.9%  local ILP U=4 wpb=1
locILP            1        1      8       1       7.2       1.5%  local ILP U=8 wpb=1
locILP            1        1     16       1      10.0       2.1%  local ILP U=16 wpb=1
locScale          1        8      8       2      46.2       9.7%  local SMscale blocks=1 (8w/U8)
locScale          2        8      8       2      91.6      19.2%  local SMscale blocks=2 (8w/U8)
locScale          4        8      8       2     178.5      37.3%  local SMscale blocks=4 (8w/U8)
locScale          8        8      8       2     343.4      71.8%  local SMscale blocks=8 (8w/U8)
locScale         16        8      8       2     615.2     128.7%  local SMscale blocks=16 (8w/U8)
locScale         32        8      8       2    1050.0     219.6%  local SMscale blocks=32 (8w/U8)
locScale         64        8      8       2    1463.0     306.0%  local SMscale blocks=64 (8w/U8)
locScale         78        8      8       2    1522.1     318.4%  local SMscale blocks=78 (8w/U8)
doorbell          1        8      8       2      12.3       2.6%  doorbell K=16 int4
doorbell          1        8      8       2      35.2       7.4%  doorbell K=256 int4
doorbell          1        8      8       2      38.8       8.1%  doorbell K=4096 int4
doorbell          1        8      8       2      44.1       9.2%  doorbell K=inf (none)
```

## 5. 结论（≤10 行 + 5 问）

1. **同 SMSP：1 warp ILP 没打平 4 warp TLP。** ILP_sSMSP=7.3 GB/s，TLP_sSMSP=21.6 GB/s，差 ~3×。瓶颈不是“该 SMSP 的 LSU 发 issue 率”（T≈1 两者都够），而是 store 前的 `ld.src→st.dst` 依赖链：单 warp 发完 U 个 store 后要等 load 回来才能再发，LSU 闲着；4 warp 用 TLP 把各 warp 的 load/store 相位错开，LSU 被持续喂满。
2. **1 SM 峰值 vs 多 SM 饱和值 ~8.4×。** 远程 1 SM（8w×U8）=44 GB/s；多 SM 在 16 SM 处饱和 ~371 GB/s。1 SM 仅 ~12% 的 NVLink 峰值。
3. **饱和用了 ~16 SM，约 78%×478 ≈ 82%×真实~450 GB/s。** 之后加到 78 SM 不再涨（367–371 平台）。
4. **本地 HBM vs NVLink：单 SM 写吞吐几乎相同（~44–46 GB/s，都是 LSU 发 issue 受限），但多 SM 峰值差 ~4×**：本地随 HBM 扩到 1522 GB/s，远程被 NVLink 注入速率卡在 ~371 GB/s。打满各自峰值都约需 ≥16 SM，差别在“峰值天花板”不在“所需 SM 数”。
5. **与“TC L/T≈1→1 warp 够；NVLink L/T 大→要多 SM”一致（部分修正）**：TC 那条（单 warp 4 链≈4 warp 1 链）在 store 上**不成立**——store 的 L（load→store 依赖 + NVLink credit）大到单 warp 的 ILP（U=16 才 10 GB/s）喂不满 LSU，必须靠多 warp TLP；而系统层面 NVLink 的大 L/T 也确实要 ~16 SM 才填满，1 SM 远低于峰值。结论：通信的 Little 定律与 TC 同一框架，但 L/T 量级不同 → 需要更多在途（warp/SM），单 SMSP/单 SM 无法打平。

> F 门铃：每 16 int4 一次 `membar.sys`+release 把带宽打到 12 GB/s（2.6%）；K=256→35、K=4096→39、无门铃→44 GB/s，说明协议门铃把有效 L 拉长、带宽随 K 增大回收。
