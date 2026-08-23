// m_p2pstore.cu  --  NVLink P2P *store* micro-benchmark (SM-issued st.global to peer)
//
// Model: Little's law on the store/LSU pipeline.
//   Issue: scheduler kicks one `st`
//   T    : initiation interval of that LSU/store pipe (cycles to kick the next st)
//   L    : how long one store occupies a slot (store-buffer / NVLink credit / doorbell)
//   in-flight needed to saturate ~ ceil(L/T)
// Tensor Core: L/T ~ 1..1.5  -> 1 warp x 4 MMA chains (ILP) ~ 4 warps x 1 chain (TLP)
// NVLink store: L/T is far larger -> need many in-flight -> many warps / many SMs.
//
// No DeepEP / NCCL / NVSHMEM. Single process, 2 GPUs, cudaDeviceEnablePeerAccess.
// GPU0 runs the kernel: ld.global local src, st.global to GPU1 peer dst.
// We never use cudaMemcpyPeer (that is the copy engine, not an SM store).
//
// Build:  nvcc -O3 -arch=sm_90 -o m_p2pstore m_p2pstore.cu
//
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

typedef int4 vec_t;                       // 16 bytes per element
static const size_t PAYLOAD_BYTES = 1UL << 28;          // 256 MB per buffer
static const size_t N_INT4         = PAYLOAD_BYTES / sizeof(vec_t); // 2^24 = 16,777,216
static const int    REPS           = 4;     // buffer rewrites inside one kernel launch
static const int    ITERS          = 3;     // timed launches
static const int    FLAG_N         = 1 << 12; // doorbell flag slots (peer device1)

#define CUDA_CHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } }while(0)

// ---- init src[i] = i (so we can verify remote store) ----
__global__ void kInit(vec_t* a, size_t n){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=s) a[i] = make_int4((int)i,(int)i,(int)i,(int)i);
}

// ---- correctness check: sample 16 int4 of dst, compare to src pattern ----
__global__ void kCheck(const vec_t* dst, size_t n, int* mismatches){
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  if(k>=16) return;
  size_t idx = (size_t)k * (n/16);
  vec_t v = dst[idx];
  vec_t e = make_int4((int)idx,(int)idx,(int)idx,(int)idx);
  if(v.x!=e.x || v.y!=e.y || v.z!=e.z || v.w!=e.w) atomicAdd(mismatches,1);
}

// ---- main store kernel ----
//   UNROLL : U, software-pipelined independent loads then stores
//   pin_mode: 0 = all warps active
//             1 = keep only warp 0            (1 warp on SMSP0)
//             2 = keep warps where (warp&3)==0 (4 warps, all on SMSP0 if mapping is round-robin)
//   doorbell_k: 0 = no doorbell; else ring a system fence+flag every K int4 written/thread
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
    // unrolled body: load U, then store U (independent -> LSU stays busy)
    for(; i + (size_t)(UNROLL-1)*stride < n; i += (size_t)UNROLL*stride){
      vec_t r[UNROLL];
      #pragma unroll
      for(int u=0;u<UNROLL;++u) r[u] = src[i + (size_t)u*stride];
      #pragma unroll
      for(int u=0;u<UNROLL;++u){
        vec_t v = r[u];
        dst[i + (size_t)u*stride] = v;                 // st.global (16B, coalesced in warp)
        if(doorbell_k){
          if(++written >= (uint64_t)doorbell_k){
            written = 0;
            __threadfence_system();                    // membar.sys: prior stores observable
            if(lane==0) flag[warpGlobal & (FLAG_N-1)] = v; // st.release.sys to peer flag
            __threadfence_system();
            __syncwarp();
          }
        }
      }
    }
    // tail (no unroll)
    for(; i<n; i+=stride){
      vec_t v = src[i];
      dst[i] = v;
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
}

template<int UNROLL>
static void launchK(int blocks, int warps_per_block, vec_t* dst, const vec_t* src,
                    int pin_mode, int doorbell_k, vec_t* flag){
  int threads = warps_per_block*32;
  kStore<UNROLL><<<blocks,threads>>>(dst, src, N_INT4, pin_mode, doorbell_k, flag);
}

static void dispatch(int unroll, int blocks, int warps_per_block, vec_t* dst,
                     const vec_t* src, int pin_mode, int doorbell_k, vec_t* flag){
  switch(unroll){
    case 1:  launchK<1 >(blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag); break;
    case 2:  launchK<2 >(blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag); break;
    case 4:  launchK<4 >(blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag); break;
    case 8:  launchK<8 >(blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag); break;
    default: launchK<16>(blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag); break;
  }
}

// time one config; returns GB/s
static double runCfg(int unroll, int blocks, int warps_per_block, vec_t* dst,
                     const vec_t* src, int pin_mode, int doorbell_k, vec_t* flag){
  cudaEvent_t e0,e1; CUDA_CHK(cudaEventCreate(&e0)); CUDA_CHK(cudaEventCreate(&e1));
  // warmup
  dispatch(unroll,blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag);
  CUDA_CHK(cudaDeviceSynchronize());
  CUDA_CHK(cudaEventRecord(e0));
  for(int it=0; it<ITERS; ++it)
    dispatch(unroll,blocks,warps_per_block,dst,src,pin_mode,doorbell_k,flag);
  CUDA_CHK(cudaEventRecord(e1)); CUDA_CHK(cudaEventSynchronize(e1));
  float ms; CUDA_CHK(cudaEventElapsedTime(&ms,e0,e1));
  CUDA_CHK(cudaEventDestroy(e0)); CUDA_CHK(cudaEventDestroy(e1));
  double bytes = (double)N_INT4 * sizeof(vec_t) * REPS * ITERS;
  return bytes / 1e9 / (ms/1000.0);
}

static int perSmsp(int warps_per_block, int pin_mode){
  if(pin_mode==1) return 1;
  if(pin_mode==2) return 4;
  int v = (warps_per_block+3)/4; return v<1?1:v;
}

int main(){
  int devs; CUDA_CHK(cudaGetDeviceCount(&devs));
  if(devs<2){ printf("need >=2 GPUs, found %d\n",devs); return 1; }

  cudaDeviceProp p; CUDA_CHK(cudaGetDeviceProperties(&p,0));
  int sms = p.multiProcessorCount;

  int can01=0; CUDA_CHK(cudaDeviceCanAccessPeer(&can01,0,1));
  printf("=== environment ===\n");
  printf("GPU0=%s GPU1=%s compute_cap=%d.%d\n", p.name, "", p.major, p.minor);
  printf("multiProcessorCount(GPU0) = %d\n", sms);
  printf("canAccessPeer 0->1 = %d\n", can01);
  printf("NVLink GPU0<->GPU1: 18 links x 26.562 GB/s = %.1f GB/s aggregate (per dir)\n", 18.0*26.562);
  double nvlink_peak = 18.0*26.562;

  // enable peer access 0 -> 1 (so SM on GPU0 can st.global to GPU1 memory)
  CUDA_CHK(cudaSetDevice(0));
  cudaError_t ea = cudaDeviceEnablePeerAccess(1,0);
  if(ea!=cudaSuccess){
    printf("cudaDeviceEnablePeerAccess(0->1) FAILED: %s\n", cudaGetErrorString(ea));
    printf("Cannot run P2P store benchmark. Stop.\n");
    return 1;
  }
  printf("peer access 0->1 enabled OK\n");

  // allocate
  size_t bufBytes = PAYLOAD_BYTES;
  vec_t *d_src, *d_dst_peer, *d_dst_local, *d_flag;
  CUDA_CHK(cudaSetDevice(0)); CUDA_CHK(cudaMalloc(&d_src, bufBytes));
  CUDA_CHK(cudaMalloc(&d_dst_local, bufBytes));
  CUDA_CHK(cudaSetDevice(1)); CUDA_CHK(cudaMalloc(&d_dst_peer, bufBytes));
  CUDA_CHK(cudaMalloc(&d_flag, FLAG_N*sizeof(vec_t)));
  CUDA_CHK(cudaMemset(d_flag, 0, FLAG_N*sizeof(vec_t)));

  // init src on GPU0
  CUDA_CHK(cudaSetDevice(0));
  kInit<<<(N_INT4+255)/256,256>>>(d_src, N_INT4);
  CUDA_CHK(cudaDeviceSynchronize());
  CUDA_CHK(cudaMemset(d_dst_peer, 0, bufBytes));   // device1
  CUDA_CHK(cudaMemset(d_dst_local, 0, bufBytes));  // device0

  // correctness check on a representative remote config (1 block, 8 warps, U=8)
  dispatch(8,1,8,d_dst_peer,d_src,0,0,d_flag);
  CUDA_CHK(cudaDeviceSynchronize());
  int* d_mm; CUDA_CHK(cudaMalloc(&d_mm,sizeof(int))); CUDA_CHK(cudaMemset(d_mm,0,sizeof(int)));
  kCheck<<<1,32>>>(d_dst_peer, N_INT4, d_mm);
  CUDA_CHK(cudaDeviceSynchronize());
  int h_mm=0; CUDA_CHK(cudaMemcpy(&h_mm,d_mm,sizeof(int),cudaMemcpyDeviceToHost));
  printf("correctness (remote 1blk/8w/U8): mismatches=%d  %s\n", h_mm, h_mm==0?"OK":"FAIL");
  CUDA_CHK(cudaFree(d_mm));
  if(h_mm!=0){
    printf("Remote store did NOT reach GPU1. Stop.\n");
    return 1;
  }

  // ===================== table =====================
  printf("\n%-14s %4s %8s %6s %7s %9s %10s  %s\n",
         "mode","sms","w/sm","unrl","smsp","GB/s","%nvlink","notes");
  printf("-------------------------------------------------------------------------------\n");

  auto emit = [&](const char* mode,int blocks,int wpb,int unroll,int pin,int doorbell,
                  vec_t* dst,const char* notes,double peak){
    int sm_used = blocks;                 // one block per SM
    int wsm = wpb;                        // warps per SM (1 block/SM)
    double g = runCfg(unroll,blocks,wpb,dst,d_src,pin,doorbell,d_flag);
    printf("%-14s %4d %8d %6d %7d %9.1f %9.1f%%  %s\n",
           mode, sm_used, wsm, unroll, perSmsp(wpb,pin), g, 100.0*g/nvlink_peak, notes);
  };

  // ---- A. kTLP (multi-warp single chain), remote ----
  for(int wpb : {1,2,4,8,16,32}){
    char note[64]; snprintf(note,sizeof(note),"kTLP warps/block=%d",wpb);
    emit("TLP",1,wpb,1,0,0,d_dst_peer,note,nvlink_peak);
  }

  // ---- B. kILP (1 warp software pipeline; also sweep warps) ----
  for(int unroll : {1,2,4,8,16}){
    char note[64]; snprintf(note,sizeof(note),"kILP U=%d wpb=1",unroll);
    emit("ILP",1,1,unroll,0,0,d_dst_peer,note,nvlink_peak);
  }
  // ILP with more warps (ILP x TLP stacking)
  for(int wpb : {4,8}){
    char note[64]; snprintf(note,sizeof(note),"kILP U=8 wpb=%d",wpb);
    emit("ILP",1,wpb,8,0,0,d_dst_peer,note,nvlink_peak);
  }

  // ---- C. same SMSP: 1 warp ILP=8  vs  4 warp TLP=1 ----
  emit("ILP_sSMSP",1,1,8,0,0,d_dst_peer,"1 warp x U=8 (SMSP0)",nvlink_peak);   // 1 warp
  emit("TLP_sSMSP",1,16,1,2,0,d_dst_peer,"4 warp(w0,4,8,12) x U=1 same SMSP",nvlink_peak);

  // ---- D. SM scaling, fixed fed-full config (8 warps/block, U=8) ----
  int smlist[] = {1,2,4,8,16,32,64,132};
  for(int s : smlist){
    int b = s; if(b>sms) b=sms;
    char note[64]; snprintf(note,sizeof(note),"D sm_scale blocks=%d (8w/U8)",b);
    emit("SMscale",b,8,8,0,0,d_dst_peer,note,nvlink_peak);
    if(b==sms) break; // don't exceed SM count
  }

  // ---- E. local HBM store control (same kTLP/kILP) ----
  for(int wpb : {1,2,4,8,16,32}){
    char note[64]; snprintf(note,sizeof(note),"local TLP wpb=%d",wpb);
    emit("locTLP",1,wpb,1,0,0,d_dst_local,note,nvlink_peak);
  }
  for(int unroll : {1,2,4,8,16}){
    char note[64]; snprintf(note,sizeof(note),"local ILP U=%d wpb=1",unroll);
    emit("locILP",1,1,unroll,0,0,d_dst_local,note,nvlink_peak);
  }
  // local SM scaling to find single-SM vs multi-SM
  for(int s : {1,2,4,8,16,32,64,132}){
    int b=s; if(b>sms) b=sms;
    char note[64]; snprintf(note,sizeof(note),"local SMscale blocks=%d (8w/U8)",b);
    emit("locScale",b,8,8,0,0,d_dst_local,note,nvlink_peak);
    if(b==sms) break;
  }

  // ---- F. doorbell dependency (remote, 8 warps/block, U=8) ----
  for(int k : {16,256,4096}){
    char note[64]; snprintf(note,sizeof(note),"doorbell K=%d int4",k);
    emit("doorbell",1,8,8,0,k,d_dst_peer,note,nvlink_peak);
  }
  emit("doorbell",1,8,8,0,0,d_dst_peer,"doorbell K=inf (none)",nvlink_peak);

  printf("\nNVLink peak reference = %.1f GB/s (18 x 26.562). %%nvlink = GB/s / peak.\n", nvlink_peak);

  CUDA_CHK(cudaSetDevice(0)); CUDA_CHK(cudaFree(d_src)); CUDA_CHK(cudaFree(d_dst_local));
  CUDA_CHK(cudaSetDevice(1)); CUDA_CHK(cudaFree(d_dst_peer)); CUDA_CHK(cudaFree(d_flag));
  printf("done.\n");
  return 0;
}
