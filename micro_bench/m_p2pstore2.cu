// m_p2pstore2.cu -- NVLink P2P *store* micro-benchmark, decision experiment for hypothesis H.
//
// Four launches, SAME total bytes written (dst of N_INT4 int4, REPS passes),
// so a 1-warp config issues the most instructions and C4 the fewest per warp.
//   C1 : 1 SM x 1 subcore x 1 warp            (A=32)
//   C2 : 1 SM x 1 subcore x 4 warp (warp 0/4/8/12 pinned same SMSP)  (A=128)
//   C3 : 1 SM x 4 subcore x 1 warp (warp 0/1/2/3)                    (A=128)
//   C4 : many SM x many warp (sweep 2..78 SM; 4 or 32 warp/block)    (A large)
// C2 and C3 have identical A and identical per-warp work; the ONLY difference is
// 1 subcore vs 4 subcores -> that is the issue-width comparison.
//
// Coverage: partition [0,n) into warp-units of W=32*U int4. Each active warp owns the
// units with residue (warp_global mod Wp). Every unit written exactly once -> same
// total bytes for all configs (n identical). No idle warp ever enters the stride.
//
// payload:  store-only (register value -> st, no ld; H's best case) and copy (ld src+st).
// unroll U = 1,4,8,16,32. C1 must sweep U to the max.
//
// Build: nvcc -O3 -arch=sm_90 -o m_p2pstore2 m_p2pstore2.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

typedef int4 vec_t;
static const size_t PAYLOAD   = 1UL << 28;             // 256 MB
static const size_t N_INT4    = PAYLOAD / sizeof(vec_t); // 2^24 = 16,777,216 (mult of 1024)
static const int    REPS      = 4;
static const int    ITERS     = 3;
static const double NVL_PEAK  = 18.0 * 26.562;          // 478.1 GB/s reference

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

// PIN: keep only warps 0,4,8,12 (same SMSP). STORE_ONLY: no ld.
template <bool PIN, int U, bool STORE_ONLY>
__global__ void kStore(vec_t* __restrict__ dst, const vec_t* __restrict__ src,
                       size_t n, int block_active_warps){
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if(PIN && (warp&3)!=0) return;                 // idle warps: never touch dst / stride
  int awarp = PIN ? (warp>>2) : warp;            // active warp index within block
  int warp_global = blockIdx.x * block_active_warps + awarp;
  size_t Wp = (size_t)gridDim.x * block_active_warps;   // total active warps
  size_t W  = 32ULL * U;                          // int4 per warp-unit
  size_t nunits = n / W;
  size_t unit = (size_t)warp_global;
  for(int rep=0; rep<REPS; ++rep){
    for(; unit<nunits; unit+=Wp){
      size_t B = unit * W;                        // base of this contiguous 32*U block
      #pragma unroll
      for(int u=0; u<U; ++u){
        size_t idx = B + (size_t)u*32 + lane;     // coalesced across the warp (lane->+l)
        vec_t v;
        if(STORE_ONLY) v = make_int4((int)idx,(int)idx,(int)idx,(int)idx);
        else           v = src[idx];
        dst[idx] = v;                                     // global store; not elidable
        // compiler barrier so the store cannot be hoisted/merged away
        asm volatile("" ::: "memory");
      }
    }
    unit = (size_t)warp_global;                   // reset for next rep
  }
}

template <bool PIN, bool STORE_ONLY>
double runU(int U, int blocks, int block_threads, int block_active_warps,
            vec_t* dst, const vec_t* src){
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
  double rel=32.0/(double)A;
  printf("%-5s %-7s %-4s %4d %4d %6d %3d %8.1f %8.1f%%  %7.4f  %s\n",
         cfg,pay,dstn,sms,nw,A,U,g, nvl?100.0*g/NVL_PEAK:0.0, rel, notes);
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

  // correctness (remote, C3 store-only U=8)
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
  // C1: 1 block,32 th,1 active warp, PIN=false
  for(int U:Ulist){ row("C1","so","nvl",1,1,32,U, runU<false,true>(U,1,32,1,d_peer,d_src), true,"1w store-only"); }
  for(int U:{1,8,32}){ row("C1","cp","nvl",1,1,32,U, runU<false,false>(U,1,32,1,d_peer,d_src), true,"1w copy"); }
  // C2: 1 block,512 th, 4 active warp(w0,4,8,12), PIN=true  (1 subcore,4 warp)
  for(int U:Ulist){ row("C2","so","nvl",1,4,128,U, runU<true ,true>(U,1,512,4,d_peer,d_src), true,"4w 1subcore so"); }
  for(int U:{1,8,32}){ row("C2","cp","nvl",1,4,128,U, runU<true ,false>(U,1,512,4,d_peer,d_src), true,"4w 1subcore cp"); }
  // C3: 1 block,128 th,4 active warp(w0,1,2,3), PIN=false (4 subcore,1 warp)
  for(int U:Ulist){ row("C3","so","nvl",1,4,128,U, runU<false,true>(U,1,128,4,d_peer,d_src), true,"4subcore so"); }
  for(int U:{1,8,32}){ row("C3","cp","nvl",1,4,128,U, runU<false,false>(U,1,128,4,d_peer,d_src), true,"4subcore cp"); }

  // C4: many SM x C3-style(4 warp/block). store-only U=8 saturation; plus U=1. 4w and 32w/block.
  int smlist[]={2,4,8,16,32,78};
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,4*b,128*b,8, runU<false,true>(8,b,128,4,d_peer,d_src), true,"4w/blk U8");
    if(b==sms)break;
  }
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,32*b,1024*b,8, runU<false,true>(8,b,1024,32,d_peer,d_src), true,"32w/blk U8");
    if(b==sms)break;
  }

  // C4 sat detail U=1 (4w/blk) at a few SM to show floor
  for(int s:{2,8,78}){ int b=s; if(b>sms)b=sms;
    row("C4","so","nvl",b,4*b,128*b,1, runU<false,true>(1,b,128,4,d_peer,d_src), true,"4w/blk U1");
  }

  // local control: store-only U=32 for C1/C2/C3/C4(78), and local C4 saturation
  row("C1","so","loc",1,1,32,32, runU<false,true>(32,1,32,1,d_loc,d_src), false,"local 1w U32");
  row("C2","so","loc",1,4,128,32, runU<true ,true>(32,1,512,4,d_loc,d_src), false,"local 4w1sub U32");
  row("C3","so","loc",1,4,128,32, runU<false,true>(32,1,128,4,d_loc,d_src), false,"local 4sub U32");
  for(int s:smlist){ int b=s; if(b>sms)b=sms;
    row("C4","so","loc",b,4*b,128*b,8, runU<false,true>(8,b,128,4,d_loc,d_src), false,"local 4w/blk U8");
    if(b==sms)break;
  }
  printf("\nNVpeak=%.1f GB/s. %%nvl only meaningful for remote. inst/w = 32/A (per-warp iterations vs C1).\n",NVL_PEAK);
  CHK(cudaSetDevice(0)); CHK(cudaFree(d_src)); CHK(cudaFree(d_loc));
  CHK(cudaSetDevice(1)); CHK(cudaFree(d_peer));
  printf("done.\n"); return 0;
}
