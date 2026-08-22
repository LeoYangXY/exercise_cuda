# 同一份 MMA + quant：一份 CUDA，对照 PTX / SASS

实验只看两个文件：

| 文件 | 干什么 |
|---|---|
| `swp_hand.cu` | **唯一内核**。`kTLP` = 同 SMSP 4 warp；`kHand` = 单 warp 4 条独立链 |
| `patch_sass.py` | 改 `kHand` 的 cubin：拆假 WAW、把 4 个 HMMA 铺开 |

平台：RTX 5050 Laptop，sm_120。`clock64`，R=4096，median of 7。  
`kTLP`：`<<<1,512>>>`，只跑 warp 0/4/8/12。  
`kHand`：`<<<1,32>>>`。  
cyc/mma = 总 cycle / (4 × rounds)，两边 MMA 总数一样。

| | cyc/mma | /TLP | 内层 SASS |
|---|---|---|---|
| kHand，ptxas 原样 | 58.1 | 1.34 | 4 个 HMMA 在 body+63/76/90/102，绕回 **75** 拍 |
| kHand，手改 cubin | **45.4** | **1.05** | HMMA 在 +24/52/80/108，间隔 27，绕回 **30** |
| kTLP 4 warp 同 SMSP | 43.2 | 1.00 | 每 warp 32 条：1×HMMA + 31 其它 |

下面三段都是 **nvcc 13.2 / nvdisasm 对当前 `swp_hand.cu` 的原文**，没有用 `…` 省略指令。SASS 里 `0.07` 是 nvdisasm 打出来的 `0.070000000298023223877`（就是 hex `0f3d8f5c29`）。

```bash
nvcc -O3 -arch=sm_120 -ptx   -o /tmp/swp_hand.ptx   micro_bench/swp_hand.cu
nvcc -O3 -arch=sm_120 -cubin -o /tmp/swp_hand.cubin micro_bench/swp_hand.cu
nvdisasm --print-code /tmp/swp_hand.cubin
python3 micro_bench/patch_sass.py /tmp/swp_hand.cubin kHand -o /tmp/kHand.patched.cubin
nvcc -O3 -arch=sm_120 -o /tmp/swp_hand micro_bench/swp_hand.cu -lcuda
/tmp/swp_hand /tmp/kHand.patched.cubin
```

立即数：

| hex | 值 | 用在哪 |
|---|---|---|
| `0f3f8020c5` / `0x3f8020c5` | 1.001 | scale |
| `0f3d8f5c29` | 0.07 | bias |
| `0f42fe0000` | 127.0 | ×127 |
| `0f3c23d70a` | 0.01 | x0 初值 |
| `0f3ca3d70a` | 0.02 | y0 初值 |
| `0f3cf5c28f` | 0.03 | z0 初值 |
| `0f3d23d70a` | 0.04 | w0 初值 |

---

## 1. 先看 CUDA：一份工作到底是什么

`kTLP` 每个 warp 每轮就干这一份。`kHand` 是把同一份复制 4 路，放进同一个 warp。

`mma` + `epi_one` 原文（`swp_hand.cu`）：

```cuda
__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3,
                                    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int epi_one(float d0, float d1, float d2, float d3) {
  float s = 1.001f, b = 0.07f;
  float v0 = fmaf(d0, s, b), v1 = fmaf(d1, s, b);
  float v2 = fmaf(d2, s, b), v3 = fmaf(d3, s, b);
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
```

`kTLP` 循环原文。`(warp & 3) != 0` 丢掉另外 12 个 warp，只留 0/4/8/12（同一个 SMSP）：

```cuda
extern "C" __global__ void kTLP(...) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if ((warp & 3) != 0) return;
  if ((warp >> 2) >= 4) return;
  // load A[4], B[2]
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
  // lane 0 写 C[warp], O[warp]
}
```

`kHand` 不走 C++ `for`。四条链 `x/y/z/w`，每条自己的 acc、B、F2I dest、sink。B 故意不同（`bx=b0+0 … bw=b0+3`），否则 ptxas 会把 4 个 `mma` CSE 成 1 个。源顺序是软件流水：

```
上一轮的 ix*  --min/max-->  clamp 到 [-128,127]  --add-->  sink
当前 acc x*   --fma 1.001+0.07-->  --max 0-->  --mul 127-->  tx*
当前 acc x*   --mma-->  覆盖 x*          （和 tx* 没有 RAW，可以重叠）
tx*           --cvt.rni-->  ix*          （给下一轮 clamp）
```

源码里 `PTX_CLAMP` / `PTX_FMA` / `PTX_MMA` / `PTX_F2I` 各写一遍 × 四条链。展开后的 PTX 就是第 4 节全文，没有省略。

---

## 2. 对照表：同一条 C++ 在三层叫什么

看 SASS 用这一张。`FMNMX … !PT` 是和 0 比取 max（relu）。`VIMNMX … 0x7f, PT` 是 min(·,127)；`VIMNMX … -0x80, !PT` 是 max(·,-128)。

| CUDA | PTX | SASS (sm_120) |
|---|---|---|
| `mma.sync.m16n8k16` | 同左 | `HMMA.16816.F32 Rd, Ra, Rb, Rc`（Rd/Ra 连续 4 个，Rb 连续 2 个） |
| `fmaf(d, 1.001, 0.07)` | `fma.rn.f32 v, d, 0f3f8020c5, 0f3d8f5c29` | `FFMA Rv, Rd, Rscale, 0.07` |
| `fmaxf(v, 0)` | `max.f32 v, v, 0f00000000` | `FMNMX Rv, RZ, Rv, !PT` |
| `v * 127` | `mul.f32 t, v, 0f42fe0000` | `FMUL Rt, Rv, 127` |
| `__float2int_rn` | `cvt.rni.s32.f32 i, t` | `F2I.NTZ Ri, Rt` |
| `min(127, i)` | `min.s32 i, i, 127` | `VIMNMX.S32 Ri, Ri, 0x7f, PT` |
| `max(-128, i)` | `max.s32 i, i, -128` | `VIMNMX.S32 Ri, Ri, -0x80, !PT` |
| `sink += i` | `add.s32 sx, sx, i` | `IADD3 Rsink, …` |
| `for` 回头 | `@p bra loop` | `BRA.U !UP0, <loop>` |

`kHand` 进循环之前 ptxas 把虚寄存器钉在这些物理寄存器上（prologue 的 `MOV R8, 0x3c23d70a` 等对得上）：

| 链 | acc 四元组 | B（故意不同） | 对应 PTX |
|---|---|---|---|
| x | **R8–R11** | **R2**（= b0） | `{x0..x3}`, `{bx, b1}` |
| y | **R12–R15** | **R24**（b0+1） | `{y0..y3}`, `{by, b1}` |
| z | **R16–R19** | **R26**（b0+2） | `{z0..z3}`, `{bz, b1}` |
| w | **R20–R23** | **R28**（b0+3） | `{w0..w3}`, `{bw, b1}` |

A 始终是 **R4–R7**。b1 在 **R3**。四条 sink 在 **R48 / R49 / R51 / R52**。  
所以看见 `HMMA R12, R4, R24, R12` 就是 y 链；`FFMA R25, R12, …` 就是 `fmaf(y0)`。

---

## 3. PTX 全文：`kTLP` 内层（`nvcc -ptx`）

C++ `for` 编成 `$L__BB0_3`。`mma` 还是 inline asm；`epi_one` 已经展开成和 C++ 一一对应的 PTX。四个 fragment 的 mul/cvt/clamp **全部写下**，没有省略。

```ptx
$L__BB0_3:
	.pragma "nounroll";
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
	    {%r46,%r45,%r44,%r43},{%r7,%r8,%r9,%r10},{%r11,%r12},{%r46,%r45,%r44,%r43};

	fma.rn.f32      %r13, %r46, 0f3F8020C5, 0f3D8F5C29;
	fma.rn.f32      %r14, %r45, 0f3F8020C5, 0f3D8F5C29;
	fma.rn.f32      %r15, %r44, 0f3F8020C5, 0f3D8F5C29;
	fma.rn.f32      %r16, %r43, 0f3F8020C5, 0f3D8F5C29;
	max.f32         %r17, %r13, 0f00000000;
	max.f32         %r18, %r14, 0f00000000;
	max.f32         %r19, %r15, 0f00000000;
	max.f32         %r20, %r16, 0f00000000;
	mul.f32         %r21, %r17, 0f42FE0000;
	cvt.rni.s32.f32 %r22, %r21;
	max.s32         %r23, %r22, -128;
	min.s32         %r24, %r23, 127;
	mul.f32         %r25, %r18, 0f42FE0000;
	cvt.rni.s32.f32 %r26, %r25;
	max.s32         %r27, %r26, -128;
	min.s32         %r28, %r27, 127;
	mul.f32         %r29, %r19, 0f42FE0000;
	cvt.rni.s32.f32 %r30, %r29;
	max.s32         %r31, %r30, -128;
	min.s32         %r32, %r31, 127;
	mul.f32         %r33, %r20, 0f42FE0000;
	cvt.rni.s32.f32 %r34, %r33;
	max.s32         %r35, %r34, -128;
	min.s32         %r36, %r35, 127;
	add.s32         %r37, %r28, %r42;
	add.s32         %r38, %r37, %r24;
	add.s32         %r39, %r38, %r32;
	add.s32         %r42, %r39, %r36;
	add.s32         %r41, %r41, 1;
	setp.ne.b32     %p3, %r41, 0;
	@%p3 bra        $L__BB0_3;
```

一条 warp、一个 HMMA、一份 epi。另外 3 个同 SMSP warp 跑同一段，硬件去重叠 stall。

---

## 4. PTX 全文：`kHand` 内层（`nvcc -ptx` 里 begin inline asm）

这就是 `asm volatile` 原样进 `.ptx` 的，从 `loop:` 到 `@p bra loop`。四条链铺开，一条没少。

```ptx
loop:
  min.s32 ix0, ix0, 127;
  max.s32 ix0, ix0, -128;
  min.s32 ix1, ix1, 127;
  max.s32 ix1, ix1, -128;
  min.s32 ix2, ix2, 127;
  max.s32 ix2, ix2, -128;
  min.s32 ix3, ix3, 127;
  max.s32 ix3, ix3, -128;
  add.s32 sx, sx, ix0;
  add.s32 sx, sx, ix1;
  add.s32 sx, sx, ix2;
  add.s32 sx, sx, ix3;
  min.s32 iy0, iy0, 127;
  max.s32 iy0, iy0, -128;
  min.s32 iy1, iy1, 127;
  max.s32 iy1, iy1, -128;
  min.s32 iy2, iy2, 127;
  max.s32 iy2, iy2, -128;
  min.s32 iy3, iy3, 127;
  max.s32 iy3, iy3, -128;
  add.s32 sy, sy, iy0;
  add.s32 sy, sy, iy1;
  add.s32 sy, sy, iy2;
  add.s32 sy, sy, iy3;
  min.s32 iz0, iz0, 127;
  max.s32 iz0, iz0, -128;
  min.s32 iz1, iz1, 127;
  max.s32 iz1, iz1, -128;
  min.s32 iz2, iz2, 127;
  max.s32 iz2, iz2, -128;
  min.s32 iz3, iz3, 127;
  max.s32 iz3, iz3, -128;
  add.s32 sz, sz, iz0;
  add.s32 sz, sz, iz1;
  add.s32 sz, sz, iz2;
  add.s32 sz, sz, iz3;
  min.s32 iw0, iw0, 127;
  max.s32 iw0, iw0, -128;
  min.s32 iw1, iw1, 127;
  max.s32 iw1, iw1, -128;
  min.s32 iw2, iw2, 127;
  max.s32 iw2, iw2, -128;
  min.s32 iw3, iw3, 127;
  max.s32 iw3, iw3, -128;
  add.s32 sw, sw, iw0;
  add.s32 sw, sw, iw1;
  add.s32 sw, sw, iw2;
  add.s32 sw, sw, iw3;
  fma.rn.f32 vx0, x0, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vx1, x1, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vx2, x2, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vx3, x3, 0f3f8020c5, 0f3d8f5c29;
  max.f32 vx0, vx0, 0f00000000;
  max.f32 vx1, vx1, 0f00000000;
  max.f32 vx2, vx2, 0f00000000;
  max.f32 vx3, vx3, 0f00000000;
  mul.f32 tx0, vx0, 0f42fe0000;
  mul.f32 tx1, vx1, 0f42fe0000;
  mul.f32 tx2, vx2, 0f42fe0000;
  mul.f32 tx3, vx3, 0f42fe0000;
  fma.rn.f32 vy0, y0, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vy1, y1, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vy2, y2, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vy3, y3, 0f3f8020c5, 0f3d8f5c29;
  max.f32 vy0, vy0, 0f00000000;
  max.f32 vy1, vy1, 0f00000000;
  max.f32 vy2, vy2, 0f00000000;
  max.f32 vy3, vy3, 0f00000000;
  mul.f32 ty0, vy0, 0f42fe0000;
  mul.f32 ty1, vy1, 0f42fe0000;
  mul.f32 ty2, vy2, 0f42fe0000;
  mul.f32 ty3, vy3, 0f42fe0000;
  fma.rn.f32 vz0, z0, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vz1, z1, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vz2, z2, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vz3, z3, 0f3f8020c5, 0f3d8f5c29;
  max.f32 vz0, vz0, 0f00000000;
  max.f32 vz1, vz1, 0f00000000;
  max.f32 vz2, vz2, 0f00000000;
  max.f32 vz3, vz3, 0f00000000;
  mul.f32 tz0, vz0, 0f42fe0000;
  mul.f32 tz1, vz1, 0f42fe0000;
  mul.f32 tz2, vz2, 0f42fe0000;
  mul.f32 tz3, vz3, 0f42fe0000;
  fma.rn.f32 vw0, w0, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vw1, w1, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vw2, w2, 0f3f8020c5, 0f3d8f5c29;
  fma.rn.f32 vw3, w3, 0f3f8020c5, 0f3d8f5c29;
  max.f32 vw0, vw0, 0f00000000;
  max.f32 vw1, vw1, 0f00000000;
  max.f32 vw2, vw2, 0f00000000;
  max.f32 vw3, vw3, 0f00000000;
  mul.f32 tw0, vw0, 0f42fe0000;
  mul.f32 tw1, vw1, 0f42fe0000;
  mul.f32 tw2, vw2, 0f42fe0000;
  mul.f32 tw3, vw3, 0f42fe0000;
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {x0,x1,x2,x3},{a0r,a1r,a2r,a3r},{bx,b1r},{x0,x1,x2,x3};
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {y0,y1,y2,y3},{a0r,a1r,a2r,a3r},{by,b1r},{y0,y1,y2,y3};
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {z0,z1,z2,z3},{a0r,a1r,a2r,a3r},{bz,b1r},{z0,z1,z2,z3};
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {w0,w1,w2,w3},{a0r,a1r,a2r,a3r},{bw,b1r},{w0,w1,w2,w3};
  cvt.rni.s32.f32 ix0, tx0;
  cvt.rni.s32.f32 ix1, tx1;
  cvt.rni.s32.f32 ix2, tx2;
  cvt.rni.s32.f32 ix3, tx3;
  cvt.rni.s32.f32 iy0, ty0;
  cvt.rni.s32.f32 iy1, ty1;
  cvt.rni.s32.f32 iy2, ty2;
  cvt.rni.s32.f32 iy3, ty3;
  cvt.rni.s32.f32 iz0, tz0;
  cvt.rni.s32.f32 iz1, tz1;
  cvt.rni.s32.f32 iz2, tz2;
  cvt.rni.s32.f32 iz3, tz3;
  cvt.rni.s32.f32 iw0, tw0;
  cvt.rni.s32.f32 iw1, tw1;
  cvt.rni.s32.f32 iw2, tw2;
  cvt.rni.s32.f32 iw3, tw3;
  add.s32 rr, rr, 1;
  setp.lt.s32 p, rr, nn;
  @p bra loop;
```

`asm volatile` 保住了相对 C++ 的语句顺序。ptxas 仍然会：重排**彼此没有 RAW** 的指令，以及把**已经死掉的临时量**涂到同一个物理寄存器。下一节就是这件事的产物。

跟一条值走三层（y1，acc 在 R13）：

```
CUDA   v1 = fmaf(d1, 1.001, 0.07);          // 在 y 链就是 fmaf(y1)
PTX    fma.rn.f32 vy1, y1, 0f3f8020c5, 0f3d8f5c29;
ptxas  FFMA R25, R13, R55, 0.07             // 和 vy0 抢 R25，所以 HMMA 插不进
手排   FFMA R81, R13, R55, 0.07             // 独立 dest，y 链 4 个 fma 一完就可以 HMMA R12
```

---

## 5. SASS 全文：`kTLP` 内层（32 条）

`nvdisasm`，循环从 `/*01d0*/` 到 BRA。左边是相对偏移 0–31。一个 warp、一个 HMMA。这一段 **也**别名（R13 写了又写）——无所谓，stall 交给另外 3 个 warp。

```
  0  /*01d0*/  HMMA.16816.F32 R8, R4, R2, R8                              // mma  acc=R8..R11  A=R4..R7  B=R2
  1  /*01e0*/  MOV R18, 0x3f8020c5                                        // MOV scale=1.001 (0x3f8020c5)
  2  /*01f0*/  UIADD3 UR4, UPT, UPT, UR4, 0x1, URZ                        // i++ (uniform)
  3  /*0200*/  UISETP.NE.U32.AND UP0, UPT, UR4, URZ, UPT                  // 循环谓词 UP0
  4  /*0210*/  @!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ                  // 收敛垫
  5  /*0220*/  FFMA R13, R8, R18, 0.07                                    // fmaf(d0) → R13
  6  /*0230*/  FMNMX R14, RZ, R13, !PT                                    // relu(d0)  FMNMX !PT = max(·,0)
  7  /*0240*/  FFMA R13, R9, R18, 0.07                                    // fmaf(d1)  WAW 复用 R13
  8  /*0250*/  FMUL R16, R14, 127                                         // d0 * 127
  9  /*0260*/  FMNMX R15, RZ, R13, !PT                                    // relu(d1)
 10  /*0270*/  FFMA R13, R10, R18.reuse, 0.07                             // fmaf(d2)
 11  /*0280*/  FFMA R14, R11, R18, 0.07                                   // fmaf(d3)
 12  /*0290*/  FMUL R15, R15, 127                                         // d1 * 127
 13  /*02a0*/  FMNMX R13, RZ, R13, !PT                                    // relu(d2)
 14  /*02b0*/  FMNMX R14, RZ, R14, !PT                                    // relu(d3)
 15  /*02c0*/  F2I.NTZ R16, R16                                           // cvt.rni d0 → int
 16  /*02d0*/  FMUL R18, R13, 127                                         // d2 * 127（R18 不再是 scale）
 17  /*02e0*/  FMUL R19, R14, 127                                         // d3 * 127
 18  /*02f0*/  F2I.NTZ R15, R15                                           // cvt.rni d1
 19  /*0300*/  VIMNMX.S32 R13, R16, -0x80, !PT                            // max(q0, -128)
 20  /*0310*/  F2I.NTZ R18, R18                                           // cvt.rni d2
 21  /*0320*/  VIMNMX.S32 R13, R13, 0x7f, PT                              // min(q0, 127)
 22  /*0330*/  VIMNMX.S32 R14, R15, -0x80, !PT                            // max(q1, -128)
 23  /*0340*/  F2I.NTZ R19, R19                                           // cvt.rni d3
 24  /*0350*/  VIMNMX.S32 R14, R14, 0x7f, PT                              // min(q1, 127)
 25  /*0360*/  VIMNMX.S32 R16, R18, -0x80, !PT                            // max(q2, -128)
 26  /*0370*/  IADD3 R12, PT, PT, R13, R14, R12                           // sink += q0+q1
 27  /*0380*/  VIMNMX.S32 R16, R16, 0x7f, PT                              // min(q2, 127)
 28  /*0390*/  VIMNMX.S32 R15, R19, -0x80, !PT                            // max(q3, -128)
 29  /*03a0*/  VIMNMX.S32 R15, R15, 0x7f, PT                              // min(q3, 127)
 30  /*03b0*/  IADD3 R12, PT, PT, R15, R12, R16                           // sink += q2+q3
 31  /*03c0*/  BRA.U UP0, `(.L_x_3)                                       // 回到 .L_x_3
```

---

## 6. SASS 全文：`kHand`，ptxas 原样（116 条）

循环从 `/*02d0*/` 到 BRA，相对偏移 0–115。四个 HMMA 在 **63 / 76 / 90 / 102**。从最后一个绕回第一个，中间 **75** 条。TC 地板是 32 拍发一条 HMMA，这里空了四十多拍。

看第 5、11、17、23 行：四个本该独立的 `fmaf(y0..y3)` 全写 **R25**。这是假 WAW。只换指令顺序、不换寄存器，调度器必须把它们串起来，HMMA 就插不进前缀。

PTX 里四个 `mma` 是挨着写的；SASS 里被拆开塞进后半段。F2I dest（R35、R36、…）倒是 16 个各不相同——PTX 的 `ix*` 跨迭代还活着，ptxas 不敢别名。

```
  0  /*02d0*/  MOV R55, 0x3f8020c5                                        // scale=1.001
  1  /*02e0*/  VIMNMX.S32 R57, R35, 0x7f, PT
  2  /*02f0*/  VIMNMX.S32 R47, R37, 0x7f, PT
  3  /*0300*/  VIMNMX.S32 R54, R39, 0x7f, PT
  4  /*0310*/  VIMNMX.S32 R46, R38, 0x7f, PT
  5  /*0320*/  FFMA R25, R12, R55.reuse, 0.07                             // fmaf(y0) dest=R25  acc y 在 R12
  6  /*0330*/  VIMNMX.S32 R56, R36, 0x7f, PT
  7  /*0340*/  VIMNMX.S32 R53, R40, 0x7f, PT
  8  /*0350*/  VIMNMX.S32 R57, R57, -0x80, !PT
  9  /*0360*/  FFMA R64, R22, R55, 0.07
 10  /*0370*/  FMNMX R27, RZ, R25, !PT
 11  /*0380*/  FFMA R25, R13, R55, 0.07                                   // fmaf(y1) 假 WAW 又写 R25
 12  /*0390*/  VIMNMX.S32 R56, R56, -0x80, !PT
 13  /*03a0*/  VIMNMX.S32 R44, R44, 0x7f, PT
 14  /*03b0*/  FMNMX R64, RZ, R64, !PT
 15  /*03c0*/  FMUL R29, R27, 127
 16  /*03d0*/  FMNMX R27, RZ, R25, !PT
 17  /*03e0*/  FFMA R25, R14, R55, 0.07                                   // fmaf(y2) 假 WAW 又写 R25
 18  /*03f0*/  IADD3 R52, PT, PT, R56, R52, R57
 19  /*0400*/  F2I.NTZ R35, R29
 20  /*0410*/  VIMNMX.S32 R56, R43, 0x7f, PT
 21  /*0420*/  FMUL R45, R27, 127
 22  /*0430*/  FMNMX R27, RZ, R25, !PT
 23  /*0440*/  FFMA R25, R15, R55.reuse, 0.07                             // fmaf(y3) 假 WAW 又写 R25
 24  /*0450*/  VIMNMX.S32 R53, R53, -0x80, !PT
 25  /*0460*/  VIMNMX.S32 R56, R56, -0x80, !PT
 26  /*0470*/  FMUL R64, R64, 127
 27  /*0480*/  FMUL R27, R27, 127
 28  /*0490*/  FFMA R29, R16, R55, 0.07
 29  /*04a0*/  FMNMX R25, RZ, R25, !PT
 30  /*04b0*/  F2I.NTZ R36, R45
 31  /*04c0*/  UIADD3 UR4, UPT, UPT, UR4, 0x1, URZ
 32  /*04d0*/  VIMNMX.S32 R47, R47, -0x80, !PT
 33  /*04e0*/  FMNMX R50, RZ, R29, !PT
 34  /*04f0*/  FFMA R29, R17, R55, 0.07
 35  /*0500*/  FMUL R25, R25, 127
 36  /*0510*/  UISETP.GE.AND UP0, UPT, UR4, UR5, UPT
 37  /*0520*/  VIMNMX.S32 R46, R46, -0x80, !PT
 38  /*0530*/  F2I.NTZ R37, R27
 39  /*0540*/  FMNMX R29, RZ, R29, !PT
 40  /*0550*/  FMUL R50, R50, 127
 41  /*0560*/  VIMNMX.S32 R45, R41, 0x7f, PT
 42  /*0570*/  IADD3 R52, PT, PT, R46, R52, R47
 43  /*0580*/  FMUL R58, R29, 127
 44  /*0590*/  F2I.NTZ R39, R50
 45  /*05a0*/  FFMA R27, R18, R55, 0.07
 46  /*05b0*/  VIMNMX.S32 R45, R45, -0x80, !PT
 47  /*05c0*/  FMNMX R29, RZ, R27, !PT
 48  /*05d0*/  FFMA R27, R19, R55, 0.07
 49  /*05e0*/  F2I.NTZ R38, R25
 50  /*05f0*/  FMUL R60, R29, 127
 51  /*0600*/  FMNMX R29, RZ, R27, !PT
 52  /*0610*/  FFMA R27, R8, R55, 0.07
 53  /*0620*/  F2I.NTZ R40, R58
 54  /*0630*/  FMUL R61, R29, 127
 55  /*0640*/  FMNMX R50, RZ, R27, !PT
 56  /*0650*/  FFMA R29, R9, R55, 0.07
 57  /*0660*/  MOV R25, R3.reuse
 58  /*0670*/  MOV R27, R3
 59  /*0680*/  FMUL R62, R50, 127
 60  /*0690*/  FMNMX R50, RZ, R29, !PT
 61  /*06a0*/  FFMA R29, R10, R55.reuse, 0.07
 62  /*06b0*/  F2I.NTZ R41, R60
 63  /*06c0*/  HMMA.16816.F32 R12, R4, R24, R12                           // 第1个 HMMA  y链 acc=R12  B=R24=b0+1  body+63
 64  /*06d0*/  VIMNMX.S32 R58, R31, 0x7f, PT
 65  /*06e0*/  FMUL R63, R50, 127
 66  /*06f0*/  VIMNMX.S32 R50, R33, 0x7f, PT
 67  /*0700*/  FFMA R33, R11, R55, 0.07
 68  /*0710*/  VIMNMX.S32 R58, R58, -0x80, !PT
 69  /*0720*/  F2I.NTZ R31, R62
 70  /*0730*/  VIMNMX.S32 R25, R42, 0x7f, PT
 71  /*0740*/  VIMNMX.S32 R60, R32, 0x7f, PT
 72  /*0750*/  VIMNMX.S32 R50, R50, -0x80, !PT
 73  /*0760*/  VIMNMX.S32 R60, R60, -0x80, !PT
 74  /*0770*/  F2I.NTZ R42, R61
 75  /*0780*/  FFMA R62, R20, R55, 0.07
 76  /*0790*/  HMMA.16816.F32 R16, R4, R26, R16                           // 第2个 HMMA  z链 acc=R16  B=R26=b0+2  间隔 13
 77  /*07a0*/  IADD3 R51, PT, PT, R60, R51, R58
 78  /*07b0*/  FMNMX R62, RZ, R62, !PT
 79  /*07c0*/  F2I.NTZ R32, R63
 80  /*07d0*/  FMNMX R61, RZ, R29, !PT
 81  /*07e0*/  MOV R29, R3
 82  /*07f0*/  VIMNMX.S32 R27, R34, 0x7f, PT
 83  /*0800*/  FMNMX R34, RZ, R33, !PT
 84  /*0810*/  FMUL R62, R62, 127
 85  /*0820*/  FMUL R61, R61, 127
 86  /*0830*/  VIMNMX.S32 R27, R27, -0x80, !PT
 87  /*0840*/  FFMA R63, R21, R55.reuse, 0.07
 88  /*0850*/  FFMA R55, R23, R55, 0.07                                   // 连 scale 寄存器 R55 都当 dest 用了
 89  /*0860*/  FMUL R34, R34, 127
 90  /*0870*/  HMMA.16816.F32 R20, R4, R28, R20                           // 第3个 HMMA  w链 acc=R20  B=R28=b0+3
 91  /*0880*/  F2I.NTZ R33, R61
 92  /*0890*/  FMNMX R63, RZ, R63, !PT
 93  /*08a0*/  FMNMX R55, RZ, R55, !PT
 94  /*08b0*/  IADD3 R51, PT, PT, R27, R51, R50
 95  /*08c0*/  FMUL R63, R63, 127
 96  /*08d0*/  VIMNMX.S32 R29, R54, -0x80, !PT
 97  /*08e0*/  VIMNMX.S32 R54, R44, -0x80, !PT
 98  /*08f0*/  FMUL R55, R55, 127
 99  /*0900*/  F2I.NTZ R34, R34
100  /*0910*/  IADD3 R29, PT, PT, R53, R48, R29
101  /*0920*/  IADD3 R56, PT, PT, R54, R49, R56
102  /*0930*/  HMMA.16816.F32 R8, R4, R2, R8                              // 第4个 HMMA  x链 acc=R8   B=R2=b0
103  /*0940*/  VIMNMX.S32 R49, R30, 0x7f, PT
104  /*0950*/  VIMNMX.S32 R53, R0, 0x7f, PT
105  /*0960*/  F2I.NTZ R43, R62
106  /*0970*/  VIMNMX.S32 R48, R25, -0x80, !PT
107  /*0980*/  VIMNMX.S32 R49, R49, -0x80, !PT
108  /*0990*/  VIMNMX.S32 R54, R53, -0x80, !PT
109  /*09a0*/  IADD3 R48, PT, PT, R48, R29, R45
110  /*09b0*/  F2I.NTZ R44, R63
111  /*09c0*/  IADD3 R49, PT, PT, R54, R56, R49
112  /*09d0*/  F2I.NTZ R30, R64
113  /*09e0*/  F2I.NTZ R0, R55
114  /*09f0*/  @!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
115  /*0a00*/  BRA.U !UP0, `(.L_x_0)                                      // 绕回 .L_x_0：距上一 HMMA 13 条 + 前缀 63 = 共 75 拍
```

---

## 7. SASS 全文：`kHand`，手排之后（还是 116 条，BRA 地址没动）

`patch_sass.py` 只做两件事：把假 WAW 的 dest 换成新物理寄存器；按真 RAW + 累加器上的真 WAR 重排。**指令一条没增删。** 地址 `/*02d0*/`–`/*0a00*/` 和 ptxas 一样，所以相对跳转不用改。

四个 HMMA 在 **24 / 52 / 80 / 108**，间隔 27，绕回 30（TC 地板 32）。第 5/11/17/23 行原来全写 R25，现在是 R25 / R81 / R86 / R89。第 24 行就可以发 `HMMA R12`。

```
  0  /*02d0*/  MOV R55, 0x3f8020c5
  1  /*02e0*/  VIMNMX.S32 R57, R35, 0x7f, PT
  2  /*02f0*/  VIMNMX.S32 R47, R37, 0x7f, PT
  3  /*0300*/  VIMNMX.S32 R54, R39, 0x7f, PT
  4  /*0310*/  VIMNMX.S32 R46, R38, 0x7f, PT
  5  /*0320*/  FFMA R25, R12, R55, 0.07                                   // fmaf(y0) dest=R25
  6  /*0330*/  VIMNMX.S32 R56, R36, 0x7f, PT
  7  /*0340*/  VIMNMX.S32 R53, R40, 0x7f, PT
  8  /*0350*/  VIMNMX.S32 R80, R57, -0x80, !PT
  9  /*0360*/  FFMA R64, R22, R55, 0.07
 10  /*0370*/  FMNMX R27, RZ, R25, !PT
 11  /*0380*/  FFMA R81, R13, R55, 0.07                                   // fmaf(y1) dest=R81 不再打 R25
 12  /*0390*/  VIMNMX.S32 R82, R56, -0x80, !PT
 13  /*03a0*/  VIMNMX.S32 R83, R44, 0x7f, PT
 14  /*03b0*/  FMNMX R84, RZ, R64, !PT
 15  /*03c0*/  FMUL R29, R27, 127
 16  /*03d0*/  FMNMX R85, RZ, R81, !PT
 17  /*03e0*/  FFMA R86, R14, R55, 0.07                                   // fmaf(y2) dest=R86
 18  /*03f0*/  IADD3 R52, PT, PT, R82, R52, R80
 19  /*0400*/  F2I.NTZ R35, R29
 20  /*0410*/  VIMNMX.S32 R87, R43, 0x7f, PT
 21  /*0420*/  FMUL R45, R85, 127
 22  /*0430*/  FMNMX R88, RZ, R86, !PT
 23  /*0440*/  FFMA R89, R15, R55, 0.07                                   // fmaf(y3) dest=R89  y链 4 个 fma 齐了
 24  /*0450*/  HMMA.16816.F32 R12, R4, R24, R12                           // 第1个 HMMA  y链 提前到 body+24
 25  /*0460*/  VIMNMX.S32 R90, R53, -0x80, !PT
 26  /*0470*/  VIMNMX.S32 R91, R87, -0x80, !PT
 27  /*0480*/  FMUL R92, R84, 127
 28  /*0490*/  FMUL R93, R88, 127
 29  /*04a0*/  FFMA R94, R16, R55, 0.07
 30  /*04b0*/  FMNMX R95, RZ, R89, !PT
 31  /*04c0*/  F2I.NTZ R36, R45
 32  /*04d0*/  UIADD3 UR4, UPT, UPT, UR4, 0x1, URZ
 33  /*04e0*/  VIMNMX.S32 R96, R47, -0x80, !PT
 34  /*04f0*/  FMNMX R50, RZ, R94, !PT
 35  /*0500*/  FFMA R97, R17, R55, 0.07
 36  /*0510*/  FMUL R98, R95, 127
 37  /*0520*/  UISETP.GE.AND UP0, UPT, UR4, UR5, UPT
 38  /*0530*/  VIMNMX.S32 R99, R46, -0x80, !PT
 39  /*0540*/  F2I.NTZ R37, R93
 40  /*0550*/  FMNMX R100, RZ, R97, !PT
 41  /*0560*/  FMUL R101, R50, 127
 42  /*0570*/  VIMNMX.S32 R102, R41, 0x7f, PT
 43  /*0580*/  IADD3 R52, PT, PT, R99, R52, R96
 44  /*0590*/  FMUL R58, R100, 127
 45  /*05a0*/  F2I.NTZ R39, R101
 46  /*05b0*/  FFMA R103, R18, R55, 0.07
 47  /*05c0*/  VIMNMX.S32 R104, R102, -0x80, !PT
 48  /*05d0*/  FMNMX R105, RZ, R103, !PT
 49  /*05e0*/  FFMA R106, R19, R55, 0.07
 50  /*05f0*/  F2I.NTZ R38, R98
 51  /*0600*/  FMUL R60, R105, 127
 52  /*0610*/  HMMA.16816.F32 R16, R4, R26, R16                           // 第2个 HMMA  z链 间隔 27
 53  /*0620*/  FMNMX R107, RZ, R106, !PT
 54  /*0630*/  FFMA R108, R8, R55, 0.07
 55  /*0640*/  F2I.NTZ R40, R58
 56  /*0650*/  FMUL R61, R107, 127
 57  /*0660*/  FMNMX R109, RZ, R108, !PT
 58  /*0670*/  FFMA R110, R9, R55, 0.07
 59  /*0680*/  MOV R111, R3
 60  /*0690*/  MOV R112, R3
 61  /*06a0*/  FMUL R62, R109, 127
 62  /*06b0*/  FMNMX R113, RZ, R110, !PT
 63  /*06c0*/  FFMA R114, R10, R55, 0.07
 64  /*06d0*/  F2I.NTZ R41, R60
 65  /*06e0*/  VIMNMX.S32 R115, R31, 0x7f, PT
 66  /*06f0*/  FMUL R63, R113, 127
 67  /*0700*/  VIMNMX.S32 R116, R33, 0x7f, PT
 68  /*0710*/  FFMA R117, R11, R55, 0.07
 69  /*0720*/  VIMNMX.S32 R118, R115, -0x80, !PT
 70  /*0730*/  F2I.NTZ R31, R62
 71  /*0740*/  VIMNMX.S32 R119, R42, 0x7f, PT
 72  /*0750*/  VIMNMX.S32 R120, R32, 0x7f, PT
 73  /*0760*/  VIMNMX.S32 R121, R116, -0x80, !PT
 74  /*0770*/  VIMNMX.S32 R122, R120, -0x80, !PT
 75  /*0780*/  F2I.NTZ R42, R61
 76  /*0790*/  FFMA R123, R20, R55, 0.07
 77  /*07a0*/  IADD3 R51, PT, PT, R122, R51, R118
 78  /*07b0*/  FMNMX R124, RZ, R123, !PT
 79  /*07c0*/  F2I.NTZ R32, R63
 80  /*07d0*/  HMMA.16816.F32 R8, R4, R2, R8                              // 第3个 HMMA  x链 间隔 27
 81  /*07e0*/  FMNMX R125, RZ, R114, !PT
 82  /*07f0*/  MOV R126, R3
 83  /*0800*/  VIMNMX.S32 R127, R34, 0x7f, PT
 84  /*0810*/  FMNMX R128, RZ, R117, !PT
 85  /*0820*/  FMUL R129, R124, 127
 86  /*0830*/  FMUL R130, R125, 127
 87  /*0840*/  VIMNMX.S32 R131, R127, -0x80, !PT
 88  /*0850*/  FFMA R132, R21, R55, 0.07
 89  /*0860*/  FFMA R133, R23, R55, 0.07
 90  /*0870*/  FMUL R134, R128, 127
 91  /*0880*/  F2I.NTZ R33, R130
 92  /*0890*/  FMNMX R135, RZ, R132, !PT
 93  /*08a0*/  FMNMX R136, RZ, R133, !PT
 94  /*08b0*/  IADD3 R51, PT, PT, R131, R51, R121
 95  /*08c0*/  FMUL R137, R135, 127
 96  /*08d0*/  VIMNMX.S32 R138, R54, -0x80, !PT
 97  /*08e0*/  VIMNMX.S32 R139, R83, -0x80, !PT
 98  /*08f0*/  FMUL R140, R136, 127
 99  /*0900*/  F2I.NTZ R34, R134
100  /*0910*/  IADD3 R141, PT, PT, R90, R48, R138
101  /*0920*/  IADD3 R142, PT, PT, R139, R49, R91
102  /*0930*/  VIMNMX.S32 R49, R30, 0x7f, PT
103  /*0940*/  VIMNMX.S32 R143, R0, 0x7f, PT
104  /*0950*/  F2I.NTZ R43, R129
105  /*0960*/  VIMNMX.S32 R48, R119, -0x80, !PT
106  /*0970*/  VIMNMX.S32 R49, R49, -0x80, !PT
107  /*0980*/  VIMNMX.S32 R144, R143, -0x80, !PT
108  /*0990*/  HMMA.16816.F32 R20, R4, R28, R20                           // 第4个 HMMA  w链 间隔 27
109  /*09a0*/  IADD3 R48, PT, PT, R48, R141, R104
110  /*09b0*/  F2I.NTZ R44, R137
111  /*09c0*/  IADD3 R49, PT, PT, R144, R142, R49
112  /*09d0*/  F2I.NTZ R30, R92
113  /*09e0*/  F2I.NTZ R0, R140
114  /*09f0*/  @!UPT UIADD3 URZ, UPT, UPT, URZ, URZ, URZ
115  /*0a00*/  BRA.U !UP0, `(.L_x_0)                                      // 绕回约 30 拍（TC 地板 32）
```

---

## 8. 手排到底动了哪种依赖

不是手写 SASS，是改 cubin 编码（sm_120 一条 16 字节）。BRA 相对偏移不能变，所以只能换序 + 换寄存器号。

| | 真依赖？ | 怎么办 |
|---|---|---|
| FFMA 读 acc，HMMA 写同一组 acc（C=D） | **真 WAR** | acc `R8–23` 钉死。该链 4 条 FFMA 完了才能 HMMA |
| `FFMA R25; FFMA R25` 两个不同 fragment | **假 WAW** | 第二条改成 R81。这就是 ILP 多出来的独立度 |
| F2I 写 R35，下轮 VIMNMX 读 R35 | **真 RAW**（跨迭代） | 不拆，最后一次写仍落 R35 |
| `sink +=` 的 IADD3 | **真 RMW** | `R48/49/51/52` 钉死 |
| A/B fragment | 循环内只读 | 钉死 |

规则：RAW 全部留着；WAW/WAR **只对钉死的寄存器**生效。没有为了 ILP 把 RAW 删掉。

cubin `REGCOUNT` 必须写宽（这里直接 255）。R78 在 count=80 时会 `illegal instruction`。

---

## 9. 这算不算极致 ILP

三层墙不要混：

| 墙 | 数 | 是什么 |
|---|---|---|
| TC 吞吐 | 32 cyc/mma | 同 SMSP 发 `HMMA.16816.F32` 的间隔 |
| 本循环发行 | 116/4 ≈ 29 | 1 条/cycle 且 F2I 全藏住时的下限 |
| TLP 实测 | 43 | 单 warp ~30 条 epi + F2I(~22) 藏不住，4 warp 重叠之后仍是 43 |

轻 FADD epi 时 ILP 已经打到 ~32，说明 4 路 HMMA 这条 ILP 是够的。重 quant 之后：假 WAW 拆掉、HMMA 铺开，45.4 vs TLP 43.2（1.05×）。再把所有 FFMA 抽到最前（“更激进的 ILP”）反而回到 ~50——F2I/clamp 扎堆，scoreboard 更疼。

做到的是 **和同 SMSP TLP 同一堵墙**。没做到教科书 modulo（每个槽 `fma[s]→HMMA[s]→F2I[s-1]→clamp[s-2]`，往 29–32 靠）。那是下一档，不是再去追 TLP 的 43。
