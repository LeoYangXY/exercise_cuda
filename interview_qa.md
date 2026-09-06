# AI Infra 面试题库（详细版）

> 分主题高频面经整理，每题三层作答结构：
> - **【口述版】**：30 秒一句话回答（面试首次应答）
> - **【详细版】**：3 分钟完整展开（追问时展开）
> - **【追问/扩展】**：可能被继续追问的点（准备好加分回答）
>
> 涵盖 200+ 道题，覆盖 AI Infra 秋招/社招常见考点。

---

## 📖 目录

1. [CUDA Kernel 基础与优化](#1-cuda-kernel-基础与优化)（30 题）
4. [分布式训练](#4-分布式训练)（20 题）
5. [通信（NCCL / NVSHMEM / RDMA）](#5-通信nccl--nvshmem--rdma)（14 题）
9. [编译器（MLIR / torch.compile / CUDA Graph）](#9-编译器mlir--torchcompile--cuda-graph)（10 题）
10. [C++ 八股](#10-c-八股)（13 题）


**分块、warp 每次 16×32、一条 wgmma 内部 4 个 core matrix，这些对。**  
右边那组 bank（row1 G0 → 8–11，row4 和 row0 撞）是按 **行距 32B 的紧凑 16×32** 算的，和红字「GM 行距 256B」矛盾，所以该打叉。真正 smem 一行是 **256B**，不拧时 8 行同一个 G 全是 **bank 0–3（8-way）**。

下面按 **warp0、第一条 wgmma（k=0..31）** 把 GM / 不拧 / 拧完画完。G = 16B = 16 个 FP8。一行 256 个 K → **G0..G15**。

---

## 1. Warp 眼里 GMEM 长什么样

GMEM 是整块 `A[128][256]`，K 连续，**行距 256B**。warp0 第一条指令只用灰的 G0、G1，但行与行之间隔着 G2..G15。

```
K →  G0  G1 | G2 G3 G4 G5 G6 G7 | G8 ...... G15
     k0-31 |     k32-127       | k128-255
             256B 一整行

r0   [G0][G1] G2 G3 ... G15
r1   [G0][G1] G2 G3 ... G15
...
r15  [G0][G1] G2 G3 ... G15     ← warp0 这 16 行
r16  ...                         warp1
...
r63  ...                         warp3 / wg0
r64-127                          wg1
```

warp 要的 16×32 **在 GM 里不连续**：`A[r][0:32]`，下一行跳 **+256B**。

---

## 2. TMA 一次搬完整个 BM×BK 吗？

**逻辑上** 一个 stage 要的 A 是整块 `128×256`。  
**硬件上** 一条 `cp.async.bulk.tensor` 的 inner box 受 swizzle 限制：

| | 一条 TMA 的 box | 次数 |
|---|---|---|
| `SWIZZLE_NONE` | inner 最大 256 元素 → 可以 **一次** `128×256` | 1 |
| `SWIZZLE_128B` | inner ≤ **128B** → 每次 `128 行 × 128 K` | **2**（k=0..127，再 k=128..255） |

128B 模式里还有更小的 **swizzle atom = 8 行 × 128B**（XOR 的单位）。一条 TMA 写的是 **整个 box**：M 上 128/8=**16 颗** atom 叠起来，K 上 **1 颗** 那么宽。所以：

```
TMA 指令（box）≠ swizzle atom（8×128B）≠ WGMMA tile（64×32）
BK=256 + 128B swizzle：2 条 TMA 才能填满 128×256 smem
```

---

## 3. 不 swizzle：smem = GM 线性，一条 wgmma

smem 一行仍 256B：`row r` 的 G0 在 `r*256`。

**warp0 第一条 wgmma** 内部 4 拍（只碰 G0/G1、行 0–15）：

```
① 行0–7  × G0     ② 行0–7  × G1
③ 行8–15 × G0     ④ 行8–15 × G1
```

**① 的 bank（行距 256B）：**

```
r0 G0: byte 0      → bank 0–3
r1 G0: byte 256    → bank 0–3
r2 G0: byte 512    → bank 0–3
...
r7 G0: 全是 bank 0–3     → 8-way
```

②：8 行 G1 全是 bank 4–7，也是 8-way。  
③④：行 8–15 的 G0 起点 `8*256=2048`，`2048/4 % 32 = 0`，**还是 bank 0–3**（下一拍，不和 ① 抢同一 cycle）。

你写的 8–11、16–19 只有「smem 里只塞了 16×32、行距 32B」才成立；TMA 搬的是整行 256B，不是那种紧凑块。

---

## 4. Swizzle 128B 之后：smem 变啥，同一条 wgmma

K=256 > 128 → smem **横着 2 列** swizzle atom（不是每行 256B 再 XOR）：

```
[所有 128 行的左半：每行 128B，8×128B atom 内 XOR]   k=0..127  = 逻辑 G0..G7
[所有 128 行的右半：同样拧]                         k=128..255 = 逻辑 G8..G15
```

**左半一颗 8×128B**（行 0–7）就是你熟悉的表，格子里是逻辑 G：

```
槽     0    1    2    3    4    5    6    7
r0     G0   G1   G2   G3   G4   G5   G6   G7
r1     G1   G0   G3   G2   G5   G4   G7   G6
...
r7     G7   G6   G5   G4   G3   G2   G1   G0
```

行 8–15 再贴一张（atom1）。warp0 第一条 wgmma 只用 **左两列 G0、G1**：

```
① 8 行逻辑 G0：物理槽 0,1,2,3,4,5,6,7 → 8 个 bank group，0-way
② 8 行逻辑 G1：槽 1,0,3,2,... 同样铺满
③④ 在行 8–15 那颗 atom 上重复
```

后面 7 条 wgmma：同一 16 行，依次吃 G2G3 … 最后两条在 **右半** atom 吃 G14G15。

WGMMA 描述符 `B128` 按这张表寻址；TMA 写的时候已经按同一张表摆好。

---

## 5. 和你这页的对照

| 你写的 | 结论 |
|---|---|
| 2 wg × K 循环 8；warp 每次 16×32 | 对 |
| 一条指令内部 4 个 8×16B | 对 |
| GM 行距 256B，不是 32B | 对，bank 必须用这个 |
| row1 G0 → bank 8–11 | 错（那是行距 32B）；应为仍是 0–3，8-way |
| TMA 一次搬完 128×256 | **仅 NONE 可以**；128B swizzle 要 **2 条 TMA**，每条 box 里再含 16 颗 8×128B atom |


提醒





可以。下面给你两个完整流程图。约定：

```text
P = TP size
C = CP size
R = EP size
A = attention head 数
D = H/A
I = MLP intermediate size
```

所有 shape 默认表示**单个 rank 上的 shape**，布局是 `[S, B, H]`。

注意：图中的 `SP` 和 `CP` 是两种 sequence 切分模式，一次运行通常选择其中一种：

```text
Megatron SP：本地 sequence = S/P
Ring CP：本地 sequence = S/C
```

---

## 流程图 1：Attention + Dense MLP

```text
┌─────────────────────────────────────────────────────────────┐
│                 Attention + Dense MLP Decoder Layer          │
└─────────────────────────────────────────────────────────────┘

输入 hidden states
        │
        ├──────────────────── SP + TP 模式 ────────────────────┐
        │                                                       │
        │  h: [S/P, B, H]                                      │
        │       │                                               │
        │       ▼                                               │
        │  LayerNorm                                             │
        │  [S/P, B, H] → [S/P, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  AllGather_SP，沿 sequence 维                        │
        │  [S/P, B, H] → [S, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  QKV Linear，Column TP                                │
        │  Wqkv: [H, 3H/P]                                      │
        │  [S, B, H] × [H, 3H/P]                                │
        │       → QKV: [S, B, 3H/P]                             │
        │       │                                               │
        │       ▼                                               │
        │  Reshape / Split Heads                                │
        │  Q,K,V: [S, B, A/P, D]                                │
        │       │                                               │
        │       ▼                                               │
        │  Local Attention                                      │
        │  QKᵀ score，逻辑 shape: [B, A/P, S, S]                │
        │  输出 O: [S, B, A/P, D]                               │
        │       │                                               │
        │       ▼                                               │
        │  Reshape                                                │
        │  [S, B, A/P, D] → [S, B, H/P]                         │
        │       │                                               │
        │       ▼                                               │
        │  Output Linear，Row TP                                │
        │  Wo: [H/P, H]                                          │
        │  [S, B, H/P] × [H/P, H]                               │
        │       → partial: [S, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  ReduceScatter_SP                                     │
        │  [S, B, H] → [S/P, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  Dropout + Residual                                   │
        │  [S/P, B, H] + [S/P, B, H]                            │
        │       → [S/P, B, H]                                    │
        │       │                                               │
        │       ▼                                               │
        │  LayerNorm                                             │
        │  [S/P, B, H] → [S/P, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  AllGather_SP                                          │
        │  [S/P, B, H] → [S, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  Dense MLP Up Projection，Column TP                   │
        │  Wup: [H, I/P]                                         │
        │  [S, B, H] × [H, I/P]                                 │
        │       → [S, B, I/P]                                    │
        │       │                                               │
        │       ▼                                               │
        │  GELU / SwiGLU                                        │
        │  [S, B, I/P] → [S, B, I/P]                            │
        │       │                                               │
        │       ▼                                               │
        │  Dense MLP Down Projection，Row TP                    │
        │  Wdown: [I/P, H]                                       │
        │  [S, B, I/P] × [I/P, H]                               │
        │       → partial: [S, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  ReduceScatter_SP                                     │
        │  [S, B, H] → [S/P, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  Dropout + Residual                                   │
        │  [S/P, B, H] → [S/P, B, H]                            │
        │                                                       │
        └───────────────────────────────────────────────────────┘


        ├──────────────────── CP + TP 模式 ─────────────────────┐
        │                                                       │
        │  h: [S/C, B, H]                                      │
        │       │                                               │
        │       ▼                                               │
        │  LayerNorm                                             │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  QKV Linear，Column TP                                │
        │  Wqkv: [H, 3H/P]                                      │
        │  [S/C, B, H] × [H, 3H/P]                              │
        │       → QKV: [S/C, B, 3H/P]                           │
        │       │                                               │
        │       ▼                                               │
        │  Reshape / Split Heads                                │
        │  Q,K,V: [S/C, B, A/P, D]                              │
        │       │                                               │
        │       ▼                                               │
        │  Ring Attention                                      │
        │  Q 本地保持: [S/C, B, A/P, D]                         │
        │  每轮 Ring P2P 发送/接收 K,V:                         │
        │      [S/C, B, A/P, D]                                 │
        │  每轮 score block: [B, A/P, S/C, S/C]                 │
        │  C 轮后输出: [S/C, B, A/P, D]                         │
        │       │                                               │
        │       ▼                                               │
        │  Reshape                                                │
        │  [S/C, B, A/P, D] → [S/C, B, H/P]                     │
        │       │                                               │
        │       ▼                                               │
        │  Output Linear，Row TP                                │
        │  [S/C, B, H/P] × [H/P, H]                             │
        │       → partial: [S/C, B, H]                          │
        │       │                                               │
        │       ▼                                               │
        │  AllReduce_TP                                         │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  Dropout + Residual                                   │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  LayerNorm                                             │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  Dense MLP Up Projection，Column TP                   │
        │  [S/C, B, H] × [H, I/P]                               │
        │       → [S/C, B, I/P]                                  │
        │       │                                               │
        │       ▼                                               │
        │  GELU / SwiGLU                                        │
        │  [S/C, B, I/P] → [S/C, B, I/P]                        │
        │       │                                               │
        │       ▼                                               │
        │  Dense MLP Down Projection，Row TP                    │
        │  [S/C, B, I/P] × [I/P, H]                             │
        │       → partial: [S/C, B, H]                           │
        │       │                                               │
        │       ▼                                               │
        │  AllReduce_TP                                         │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  Dropout + Residual                                   │
        │  [S/C, B, H] → [S/C, B, H]                            │
        │                                                       │
        └───────────────────────────────────────────────────────┘
```

Dense Decoder 中：

```text
TP：
  QKV 按 head 切
  Attention output projection 按 hidden 输入维切
  MLP up 按 intermediate 切
  MLP down 按 intermediate 输入切

SP：
  激活的 sequence 维为 S/P
  Column Linear 前 AllGather
  Row Linear 后 ReduceScatter

CP：
  Attention 的 sequence 维为 S/C
  Ring 传递 K/V
  不需要把完整 S 一次性 AllGather
```

---

## 流程图 2：Attention + MoE MLP

Attention 部分仍然是上面的 Attention；区别是 MLP 换成了 Router + Expert。

```text
┌─────────────────────────────────────────────────────────────┐
│                    Attention + MoE Decoder Layer              │
└─────────────────────────────────────────────────────────────┘

输入 hidden states
        │
        ├────────────── SP + TP Attention ─────────────────────┐
        │                                                       │
        │  h: [S/P, B, H]                                      │
        │       │                                               │
        │       ▼                                               │
        │  LayerNorm                                             │
        │  [S/P, B, H] → [S/P, B, H]                            │
        │       │                                               │
        │       ▼                                               │
        │  AllGather_SP                                         │
        │  [S/P, B, H] → [S, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  QKV Column TP                                        │
        │  [S, B, H] × [H, 3H/P]                               │
        │       → [S, B, 3H/P]                                  │
        │       → Q,K,V: [S, B, A/P, D]                         │
        │       │                                               │
        │       ▼                                               │
        │  Local Attention                                      │
        │  score: [B, A/P, S, S]                               │
        │  output: [S, B, A/P, D]                               │
        │       │                                               │
        │       ▼                                               │
        │  Output Row TP                                        │
        │  [S, B, H/P] → partial [S, B, H]                      │
        │       │                                               │
        │       ▼                                               │
        │  ReduceScatter_SP                                     │
        │  [S, B, H] → [S/P, B, H]                              │
        │       │                                               │
        │       ▼                                               │
        │  Dropout + Residual                                   │
        │  [S/P, B, H] → [S/P, B, H]                            │
        │                                                       │
        └───────────────────────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────────┐
        │                  MoE MLP：SP + EP                      │
        └────────────────────────────────────────────────────────┘

        x: [S/P, B, H]
        │
        ▼
    LayerNorm
    [S/P, B, H] → [S/P, B, H]
        │
        ▼
    Flatten Token
    [S/P, B, H] → [T, H]

    T = B × S/P
        │
        ▼
    Router Linear
    Wrouter: [H, E]
    [T, H] × [H, E]
        → logits: [T, E]
        │
        ▼
    Top-k
    logits: [T, E]
        → expert_ids: [T, K]
        → gate_scores: [T, K]
        │
        ▼
    Token Permute / Expand
    [T, H] → [T×K, H]
        │
        ▼
    按目标 EP rank 分桶

    发往 rank j:
    [n(i→j), H]

        │
        ▼
    AlltoAll_EP：Dispatch
    每个 rank 发送:
    [n(i→0), H], [n(i→1), H], ..., [n(i→R-1), H]

    每个 rank 接收:
    [Nrecv, H]

    Nrecv = 所有 source rank 发来的 token 副本数
        │
        ▼
    按本地 expert 分组

    Expert e 输入:
    [ne, H]

    当前 EP rank 只保存 E/R 个 expert
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        │  Expert 内部不使用 TP                        │
        │  [ne, H]                                    │
        │      │                                      │
        │      ▼                                      │
        │  Up: [ne, H] → [ne, I]                      │
        │      │                                      │
        │      ▼                                      │
        │  Activation: [ne, I] → [ne, I]              │
        │      │                                      │
        │      ▼                                      │
        │  Down: [ne, I] → [ne, H]                    │
        │                                             │
        └─────────────────────────────────────────────┘

        或者

        ┌─────────────────────────────────────────────┐
        │  Expert 内部继续使用 TP_e                    │
        │  [ne, H]                                     │
        │      │                                       │
        │      ▼                                       │
        │  Up Column TP_e                              │
        │  [ne, H] × [H, I/TP_e]                       │
        │      → [ne, I/TP_e]                          │
        │      │                                       │
        │      ▼                                       │
        │  Activation                                   │
        │  [ne, I/TP_e] → [ne, I/TP_e]                 │
        │      │                                       │
        │      ▼                                       │
        │  Down Row TP_e                                │
        │  [ne, I/TP_e] × [I/TP_e, H]                  │
        │      → partial [ne, H]                       │
        │      │                                       │
        │      ▼                                       │
        │  AllReduce_TP_e                              │
        │  partial [ne, H] → [ne, H]                   │
        │                                             │
        └─────────────────────────────────────────────┘
        │
        ▼
    AlltoAll_EP：Combine
    [Nrecv, H] → [T×K, H]
        │
        ▼
    Unpermute + Gate Weighted Sum
    [T×K, H] + gate_scores [T, K]
        → [T, H]
        │
        ▼
    Reshape
    [T, H] → [S/P, B, H]
        │
        ▼
    Dropout + Residual
    [S/P, B, H] → [S/P, B, H]

        │
        ▼
    Decoder Layer 输出
    [S/P, B, H]
```

如果使用 **CP + EP**，MoE 部分只需要把本地 sequence 换成 `S/C`：

```text
x: [S/C, B, H]
Flatten: [S/C, B, H] → [T, H]

T = B × S/C

Router:
[T, H] → [T, E] → ids/scores [T, K]

EP AlltoAll:
[n(i→j), H] → [Nrecv, H]

Local Expert:
[ne, H] → [ne, I] → [ne, H]

Combine:
[Nrecv, H] → [T×K, H] → [T, H]
                  → [S/C, B, H]
```

最后对应关系就是：

```text
Dense Decoder:
  Attention = TP + SP 或 TP + CP
  MLP       = TP + SP 或 TP + CP
  EP        = 不存在

MoE Decoder:
  Attention = TP + SP 或 TP + CP
  MLP       = EP +（可选 Expert TP）+ SP/CP
```

其中：

```text
SP：AllGather / ReduceScatter，处理 activation 的 sequence 维
TP：Column Linear / Row Linear，处理 head、hidden、intermediate 维
EP：AlltoAll，处理 token 到 expert 的重新分布
CP：Ring P2P 或 Ulysses AlltoAll，处理 Attention 的 sequence 维
```





---

# 1. CUDA Kernel 基础与优化

---

## 1.19 CUDA 的 atomic 操作原理？`atomicCAS` 怎么实现任意原子？

**【口述版】**
Atomic 由 L2 cache 上的专用硬件单元执行，对同一地址的多个请求会被合并/串行化。`atomicCAS` （compare-and-swap）是最通用的原子原语，可以用它 loop 实现任意原子操作（如 `atomicMax` for float、`atomicAdd` for double pre-Pascal）。

**【详细版】**

**`atomicCAS` 用法**：
```cpp
__device__ float atomicMax_float(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float new_val = fmaxf(val, __int_as_float(assumed));
        old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}
```
- `atomicCAS(addr, expected, new_val)`：如果 `*addr == expected` 则写 `new_val`，返回旧值；否则不写，返回当前值。
- Loop 重试直到 CAS 成功。

**性能注意**：
- 同地址 atomic 争用高时会严重退化。
- 优化：**warp 内先 reduce 再 atomic**（32 个线程变 1 次 atomic），或者 **每 block reduce 后只由 tid==0 做 atomic**。

**【追问/扩展】**
- **硬件原子 vs CAS 软实现**：`atomicAdd`（int32/fp32）、`atomicMax`（int）等有硬件 fast path；其他类型要 CAS loop。
- **Pascal 之前没有 `atomicAdd(float*)` for fp64**：要用 CAS 模拟。
- **System-wide atomic**（`atomicAdd_system`）：跨 GPU / CPU 地址，Volta+ 支持。
- **L2 atomic vs SMEM atomic**：SMEM 有 per-bank atomic 单元，延迟低。

---

# 4. 分布式训练

---

【仔细】 ## 4.11 Sequence Parallelism 的原理？

**【口述版】**
Sequence Parallelism 把 sequence 维度切分到多卡，主要有两个语境：一是 Megatron 的 SP，在 LayerNorm/Dropout 处沿 sequence 切分配合 TP 减少激活值冗余；二是 Ring Attention 类的长序列 SP，把整条序列切分到多卡做分布式 Attention 计算。

**【详细版】**

**Megatron-LM Sequence Parallelism**：

在 TP 中，每层的 AllReduce 会在每卡产生完整的激活值（batch × seq × hidden），这部分是冗余的。SP 的改进：

```
标准 TP：
[Full Activation] → LayerNorm → [Full] → TP Linear → [Partial] → AllReduce → [Full]

加 SP 后：
[Split Activation along seq] → LayerNorm → [Split] → AllGather → [Full] → TP Linear → [Partial] → ReduceScatter → [Split]
```

- 将 AllReduce 拆分为 AllGather + ReduceScatter
- LayerNorm/Dropout 在切分后的 activation 上计算（每卡只处理 seq/TP 长度）
- 激活值显存减少 TP 倍（从 batch×seq×hidden 降到 batch×seq/TP×hidden）

**通信量分析**：
- 标准 TP：每层 2 × AllReduce，AllReduce 通信量 = 2 × (N-1)/N × M
- SP + TP：每层 2 × AllGather + 2 × ReduceScatter，总通信量 = 2 × (N-1)/N × M（不变！）
- SP 的通信量和标准 TP 完全相同，但激活值显存降低 TP 倍

**DeepSpeed Ulysses（另一种 Sequence Parallelism）**：
```
输入: [batch, seq, hidden] → 沿 seq 切分到 N 卡
Q,K,V 投影: 每卡各自计算 [batch, seq/N, hidden]
AlltoAll: 重新分布为 [batch, seq, hidden/N]（每卡有完整 seq 但部分 head）
Attention: 每卡独立计算部分 head 的 attention（需要完整 seq）
AlltoAll: 恢复为 [batch, seq/N, hidden]
```
- 使用 AlltoAll 代替 AllReduce
- 通信量：2 × AlltoAll = 2 × (N-1)/N × M（和 TP 相同）
- 优势：每卡处理完整 seq 的部分 head，attention 计算无需额外通信

**【追问/扩展】**
- **Megatron SP vs Ulysses SP vs Ring Attention**：Megatron SP 配合 TP 减少激活值显存；Ulysses 用 AlltoAll 做 head 维度重分布；Ring Attention 用环形 P2P 传 KV 做超长序列。
- **为什么 SP 不增加通信量**：AllReduce = AllGather + ReduceScatter，SP 只是把这两步拆开，在中间插入 LayerNorm/Dropout。
- **长序列训练的选择**：seq < 8K 用标准 TP + SP；8K~128K 用 Ulysses；128K+ 用 Ring Attention。

---

【随意】 ## 4.12 Expert Parallelism（MoE 的分布式训练）？

**【口述版】**
Expert Parallelism 把 MoE 层的不同 expert 放在不同卡上。Router 决定每个 token 去哪些 expert 后，用 AlltoAll 通信把 token 发到对应卡上计算，算完再 AlltoAll 发回来。核心挑战是负载均衡和通信效率。

**【详细版】**

**MoE（Mixture of Experts）基本结构**：
```
Input → Router(x) → top-k experts → Weighted Sum → Output

Router 输出: gate_scores = softmax(x @ W_gate)  [tokens, num_experts]
top-k 选择: 每个 token 选 k 个 expert（通常 k=1 或 2）
```

**Expert Parallelism 的分布方式**：
- 假设 E 个 expert，EP_size 张卡做 Expert Parallelism
- 每卡持有 E / EP_size 个 expert
- Non-expert 层（Attention、LayerNorm）在各卡上复制

**通信流程（每个 MoE 层）**：
```
Step 1: 各卡 Router 计算 → 确定每个 token 去哪些 expert
Step 2: AlltoAll dispatch → 把 token 发给持有对应 expert 的卡
        [local_tokens, hidden] → [expert_tokens, hidden]（按 expert 重新分布）
Step 3: 各卡并行计算自己的 expert
Step 4: AlltoAll combine → 把结果发回原来的卡
        [expert_tokens, hidden] → [local_tokens, hidden]
Step 5: 加权求和
```

**负载均衡问题**：
- 如果所有 token 都路由到同一个 expert → 该卡计算量爆炸，其他卡空闲
- 解决方案：
  - **Auxiliary Load Balancing Loss**：加一个辅助 loss 鼓励均匀路由
  - **Expert Capacity**：每个 expert 设容量上限，超出的 token 被丢弃或 overflow 到其他 expert
  - **Token dropping**：超出容量的 token 直接跳过 MoE 层

```python
# 辅助负载均衡 loss（简化）
# f_i: 分配到 expert i 的 token 比例
# P_i: router 对 expert i 的平均概率
aux_loss = num_experts * sum(f_i * P_i for i in range(num_experts))
total_loss = task_loss + alpha * aux_loss  # alpha 通常 0.01
```

**EP 与其他并行的组合**：
- **EP + DP**：non-expert 层用 DP，expert 层用 EP
- **EP + TP**：每个 expert 内部还可以做 TP
- **EP + PP**：MoE 层和 Dense 层分到不同的 pipeline stage
- **Megablocks**：将不同 expert 的计算打包成一个大矩阵乘（block-sparse），避免 load imbalance

**通信量分析**：
- 每个 MoE 层：2 × AlltoAll
- AlltoAll 通信量 = (EP-1)/EP × tokens × hidden × 2 bytes × top_k
- 对比 TP 的 AllReduce：AlltoAll 的数据量通常更小（只传被路由的 token）

**【追问/扩展】**
- **Expert Capacity Factor**：通常设为 1.0~1.5，表示每个 expert 处理的 token 数为平均值的 1.0~1.5 倍。
- **DeepSpeed-MoE**：提供了高效的 MoE 实现，包括 Hierarchical AlltoAll（机内先 AlltoAll 再跨机）。
- **Mixtral 8x7B**：每层 8 个 expert，每 token 选 2 个，实际激活参数只有 ~13B。
- **GShard vs Switch Transformer**：GShard 用 top-2 routing，Switch 用 top-1（更高效，精度也够）。

---

---



**LL128 协议细节**：
```
每个传输单元：
┌───────────────────────────────────┬──────────┐
│           120B data               │  8B flag │   = 128 Bytes
└───────────────────────────────────┴──────────┘

利用 GPU 128B cache line 原子性：
  - 整个 128B 要么全部可见，要么全部不可见
  - 接收端读 128B，检查最后 8B flag
  - flag 正确 → 前 120B 数据有效

带宽利用率: 120/128 = 93.75%
延迟: 接近 LL
适用: NVLink 场景效果最好（NVLink 保证 128B 原子性）
```

**Simple 协议细节**：
```
使用 ring buffer + head/tail pointer:

生产者                          消费者
   │    ┌──────────────────┐    │
   ├──→ │   Chunk 0        │ ──→┤
   │    │   Chunk 1        │    │
   │    │   Chunk 2        │    │
   │    │   ...            │    │
   │    └──────────────────┘    │
   │                            │
   └── tail ptr    head ptr ────┘

  生产者写完 chunk 后更新 tail
  消费者发现 head < tail 时读取并处理
  需要 memory fence 保证 ordering

  buffer 大小可配置（NCCL_BUFFSIZE，默认 4MB）
  proxy thread 负责 host↔device 数据搬运（网络场景）

**性能实测（8xA100 NVLink AllReduce）**：
```
Message Size    Protocol    Latency     Bus BW
64 B            LL          ~8 μs       ~0.01 GB/s
4 KB            LL128       ~12 μs      ~0.3 GB/s
256 KB          LL128       ~25 μs      ~10 GB/s
16 MB           Simple      ~120 μs     ~130 GB/s
1 GB            Simple      ~3.5 ms     ~280 GB/s
```

**【追问/扩展】**
- **为什么 LL128 在 NVLink 上效果好**：NVLink 支持 128B 原子写入，PCIe 不保证，所以 PCIe 场景 LL128 可能退化为 LL。
- **NCCL_BUFFSIZE**：Simple 协议的 buffer 大小，增大可以提高吞吐但占用更多 GPU 显存。
- **proxy thread**：Simple 协议跨节点时需要 CPU proxy 线程负责 IB verbs 调用，这是一个潜在的 CPU 瓶颈点。NCCL 2.19+ 引入了 kernel-initiated 通信来绕过 proxy。
- **GDR + LL**：开启 GPUDirect RDMA 后 LL 协议可以直接从 GPU 内存轮询，更低延迟。



**3D 并行中的通信分析（Megatron-LM 风格）**：

| 并行维度 | 通信操作 | 通信量/step | 频率 | 推荐路径 |
|---|---|---|---|---|
| TP | AllReduce | 2 × act_size × layers | 每层 2 次 | NVLink |
| PP | P2P Send/Recv | micro_batch × act_size | 每 micro-batch | IB |
| DP | AllReduce/RS+AG | 2 × model_size | 每 step 1 次 | IB |
| CP (Context) | AllGather + P2P | seq_len × hidden | 每层 | NVLink/IB |

**【追问/扩展】**
- **HSDP 的拓扑映射**：节点内 FSDP shard（利用 NVLink 高带宽做 AllGather），节点间 DDP replicate（只需 AllReduce 梯度）。
- **NCCL_TOPO_FILE**：可以手动指定拓扑 XML 文件覆盖自动检测（调试或特殊硬件）。
- **Cross-node NVLink（NVL72）**：GB200 NVL72 打破了节点内/节点间的界限，72 GPU 全部 NVLink，TP 可以扩展到 72。
- **Rail-optimized 拓扑**：每个 GPU 的网卡走独立的 rail（交换机），避免 incast，详见 5.11。

---

---

【跳过】 ## 5.10 通信带宽的理论分析？Bus bandwidth vs Algorithm bandwidth？

**【口述版】**
Algorithm bandwidth = 数据量 / 时间，衡量对应用有效的吞吐；Bus bandwidth = Algorithm bandwidth × 校正因子，衡量链路实际利用率。例如 Ring AllReduce 的校正因子是 `2(N-1)/N`，因为每份数据实际在链路上传了 `2(N-1)/N` 倍。Bus bandwidth 更适合评估是否打满了硬件带宽。

**【详细版】**

**两个带宽指标的定义**：

```
Algorithm Bandwidth = S / t
  S = 操作的数据量（如 AllReduce 的 tensor size）
  t = 操作的端到端时间

Bus Bandwidth = Algorithm Bandwidth × correction_factor
  correction_factor 取决于算法和 GPU 数量 N
```

**各操作的校正因子**：

| 操作 | 理想 Algorithm BW | 校正因子 | Bus BW |
|---|---|---|---|
| AllReduce (Ring) | S/t | 2(N-1)/N | S/t × 2(N-1)/N |
| ReduceScatter | S/t | (N-1)/N | S/t × (N-1)/N |
| AllGather | S/t | (N-1)/N | S/t × (N-1)/N |
| Broadcast | S/t | 1 | S/t |
| Reduce | S/t | 1 | S/t |
| AllToAll | S/t | (N-1)/N | S/t × (N-1)/N |

**推导过程（Ring AllReduce）**：
```
Ring AllReduce = ReduceScatter + AllGather

ReduceScatter:
  - N 个 GPU 排成环
  - 数据分 N 份，每份 S/N
  - N-1 步，每步每 GPU 发送 S/N
  - 每 GPU 总发送: (N-1) × S/N
  
AllGather:
  - 同样 N-1 步，每步 S/N
  - 每 GPU 总发送: (N-1) × S/N

总计每 GPU 发送: 2(N-1) × S/N

Bus Bandwidth = 每 GPU 总发送量 / 时间
             = 2(N-1)(S/N) / t
             = (S/t) × 2(N-1)/N
```

**实际例子**：
```
场景: 8×H100 NVSwitch, AllReduce 1GB FP16
NVLink 单方向带宽: 450 GB/s

测量:
  t = 2.5 ms

Algorithm BW = 1 GB / 2.5 ms = 400 GB/s
Bus BW = 400 × 2(8-1)/8 = 400 × 1.75 = 700 GB/s

理论峰值 Bus BW = 450 GB/s（单向）或 900 GB/s（双向）
实际利用率 = 700/900 = 77.8% ← 相当不错

注意: 
  N=8 时校正因子 = 2×7/8 = 1.75
  N=∞ 时校正因子 → 2（极限值）
```

**nccl-tests 输出解读**：
```bash
$ ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 8

#       size    count   type  redop  time    algbw     busbw
       8(B)        2  float    sum  8.32   0.00      0.00
    1024(B)      256  float    sum  10.5   0.10      0.17
    1(MB)    262144  float    sum  22.1   47.5      83.1
    1(GB)  268435456 float    sum  2812   381.4     667.0

# algbw = size / time (GB/s)
# busbw = algbw × 2(N-1)/N (GB/s)
# busbw 越接近硬件峰值，说明通信越高效
```

**带宽模型用于预测训练耗时**：
```
AllReduce 时间估算:
  t_allreduce = latency + S × 2(N-1)/(N × BW)

其中:
  latency = α（启动延迟，通常 5-20 μs）
  S = 数据量
  N = GPU 数
  BW = 单链路带宽

例: 175B 模型，FP16 梯度 = 350GB
    1024 GPU, IB NDR 400Gbps = 50 GB/s (每 GPU 8 NIC)
    
    若 DP=1024 (no TP/PP):
    t = 20μs + 350GB × 2×1023/1024 / (8×50 GB/s)
    ≈ 20μs + 700GB / 400 GB/s
    ≈ 1.75 s  ← 太慢！
    
    所以需要 TP+PP 减少 DP 维度的通信量
```

**【追问/扩展】**
- **为什么 Bus BW 可能超过单方向带宽**：NVSwitch 全互联时，环上相邻两个 GPU 走不同 NVSwitch 端口，相当于每个 GPU 同时发送和接收走不同物理链路。
- **小消息瓶颈是延迟不是带宽**：`t = α + S/BW`，当 S 很小时 α 主导，此时 Bus BW 无意义。
- **有效带宽受限于最慢链路**：异构网络中（节点内 NVLink + 节点间 IB），Ring 的带宽被最慢的那条 IB 链路限制。
- **nccl-tests 的使用**：是评估集群通信性能的标准工具，面试中提到说明你有实操经验。

---

**策略 1: DDP Gradient Bucketing**：
```
DDP 反向传播:

Layer N (最后一层):
  ┌────────┐
  │Backward│
  └───┬────┘
      ↓ grad ready
  ┌──────────────────┐
  │ Bucket 0 梯度    │→ AllReduce (同时)
  └──────────────────┘        ↑
                              │ overlap
Layer N-1:                    ↓
  ┌────────┐         ┌──────────────┐
  │Backward│         │ AllReduce    │
  └───┬────┘         │ Bucket 0     │
      ↓              └──────────────┘
  ┌──────────────────┐
  │ Bucket 1 梯度    │→ AllReduce (同时)
  └──────────────────┘

Layer N-2:
  ┌────────┐         ┌──────────────┐
  │Backward│         │ AllReduce    │
  └───┬────┘         │ Bucket 1     │
  ...               └──────────────┘

Bucket 大小 (默认 25MB) 的 trade-off:
  太大 → overlap 窗口小，要等久才能开始通信
  太小 → 太多 AllReduce 调用，launch 开销大
```

**策略 2: FSDP Forward/Backward Prefetch**：
```
FSDP Forward:
  ┌──────────────────────────────────────────────┐
  │ AllGather(L0) → Compute(L0) → Free(L0)      │
  │         ┌── AllGather(L1) → Compute(L1) →    │
  │         │        ┌── AllGather(L2) → ...     │
  │         │        │                           │
  │ prefetch overlap: 在 compute(Li) 时          │
  │ 提前 AllGather(Li+1)                         │
  └──────────────────────────────────────────────┘

  forward_prefetch=True:
    Compute(Li) 开始时就发起 AllGather(Li+1)

FSDP Backward:
  BACKWARD_PRE:  compute(Li) 开始前 AllGather(Li-1)
  BACKWARD_POST: compute(Li) 结束后 AllGather(Li-1)
  推荐 BACKWARD_PRE，overlap 更充分
```

**策略 3: Tensor Parallelism Overlap**：
```
Megatron-LM column parallel + row parallel:

  输入 X
    ↓
  [AllGather X] ← 可以和上一层的 output 后处理 overlap
    ↓
  f(X×W_col)    ← 计算
    ↓
  [ReduceScatter] ← 可以和 g(output) overlap
    ↓
  输出 Y

  具体做法:
  1. 将 W_col 计算拆成多块
  2. 第一块计算完就开始 AllReduce
  3. 同时计算后续块
  
  Megatron-LM 的 --overlap-grad-reduce 和 --overlap-param-gather
```

**策略 4: Pipeline Parallelism 1F1B + Interleaving**：
```
1F1B Schedule:
  Stage 0: F F F F B B B B  (先 forward 再 backward)
  Stage 1: . F F F F B B B B
  
  空闲（bubble）无法有效利用

Interleaved 1F1B (Megatron):
  每个 stage 处理多个 virtual stages
  减少 bubble + 通信和计算 overlap:
  
  Stage 0: F0 B3 F0 B3 F0 B3  (在 F 和 B 之间穿插不同 chunk)
  通信(Send/Recv)发生在 chunk 切换时，和下一个 chunk 的计算 overlap
```

**实现细节 — CUDA Stream 管理**：
```python
# 典型的 overlap 实现
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

for layer in layers:
    with torch.cuda.stream(compute_stream):
        output = layer.forward(input)
    
    # 通信依赖计算完成
    comm_stream.wait_stream(compute_stream)
    
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(output.grad, async_op=True)
    
    # 下一层计算不依赖当前层通信
    # → 自然 overlap
```

**Overlap 的挑战**：
```
1. SM 资源竞争:
   NCCL kernel 占用 SM → 计算 kernel 可用 SM 减少
   解决: NCCL 用少量 SM（通常 1-2 个 channel 用 ~16 SM）
         CUDA_DEVICE_MAX_CONNECTIONS 增大 stream 并行度

2. PCIe/NVLink 带宽竞争:
   通信和计算（如 HBM 访问）可能竞争总线
   NVLink 和 HBM 独立通道，竞争较小

3. 依赖链管理:
   event/stream 同步过多会破坏 overlap
   过少会导致数据竞争

4. CUDA_DEVICE_MAX_CONNECTIONS:
   默认 8 个 hardware queue
   增大到 32 可以让更多 stream 真正并行
   但每个 queue 占用 SM 资源

环境变量:
  CUDA_DEVICE_MAX_CONNECTIONS=32
  NCCL_MAX_NCHANNELS=2   # 限制 NCCL SM 占用
```

**【追问/扩展】**
- **Overlap 效率量化**：`overlap_ratio = 1 - T_total / (T_compute + T_comm)`，理想 = `min(T_comm, T_compute) / (T_compute + T_comm)`。
- **nsys 分析 overlap**：用 nsys 看 NCCL kernel 和 compute kernel 的时间线重叠情况。
- **torch.distributed 的 async_op**：`dist.all_reduce(tensor, async_op=True)` 返回 `Work` 对象，`.wait()` 时才阻塞。
- **Flux / CoCoNet**：自动化 overlap 的研究工作，通过编译优化自动拆分计算和通信实现 overlap。

---


# 10. C++ 八股

【仔细】 ## 10.1 C++ 智能指针（unique_ptr / shared_ptr / weak_ptr）？

**【口述版】**
`unique_ptr` 独占所有权，不能拷贝只能 move，零开销；`shared_ptr` 共享所有权，用引用计数管理生命周期，引用计数归零时自动析构；`weak_ptr` 不增加引用计数，解决 `shared_ptr` 循环引用问题。CUDA 项目中常用 `unique_ptr` 管理设备内存（自定义 deleter 调用 `cudaFree`）。

**【详细版】**

**`std::unique_ptr<T>`**：
```cpp
auto p = std::make_unique<int>(42);
// auto p2 = p;         // 编译错误：不能拷贝
auto p2 = std::move(p); // OK：转移所有权，p 变成 nullptr
```
- 零开销抽象：大小 = 裸指针大小（无 deleter 时）
- 支持自定义 deleter：
```cpp
struct CudaDeleter {
    void operator()(void* ptr) { cudaFree(ptr); }
};
using CudaUniquePtr = std::unique_ptr<void, CudaDeleter>;

void* raw;
cudaMalloc(&raw, size);
CudaUniquePtr gpu_mem(raw);  // 离开作用域自动 cudaFree
```
- 自定义 deleter 会影响 `unique_ptr` 的大小（如果 deleter 是有状态的）

**`std::shared_ptr<T>`**：
```cpp
auto p1 = std::make_shared<int>(42);
auto p2 = p1;  // 引用计数 +1（现在为 2）
p1.reset();     // 引用计数 -1（现在为 1）
// p2 离开作用域 → 引用计数归零 → 析构
```
- 内部有两个指针：指向对象的指针 + 指向控制块的指针
- 控制块包含：strong count、weak count、deleter、allocator
- `make_shared` 一次分配（对象 + 控制块连续），减少内存碎片和 cache miss
- **线程安全**：引用计数的增减是原子操作，但对象本身的访问不是线程安全的
- 开销：每次拷贝/析构都要原子操作引用计数

**`std::weak_ptr<T>`**：
```cpp
std::shared_ptr<Node> a = std::make_shared<Node>();
std::shared_ptr<Node> b = std::make_shared<Node>();
a->next = b;   // shared_ptr: 循环引用！
b->prev = a;   // 如果用 shared_ptr，a 和 b 永远不会被释放

// 解决：b->prev 用 weak_ptr
b->prev = std::weak_ptr<Node>(a);

// 使用 weak_ptr
if (auto locked = b->prev.lock()) {
    // locked 是 shared_ptr，对象还存在
} else {
    // 对象已被析构
}
```
- 不增加 strong count，只增加 weak count
- `lock()` 返回 `shared_ptr`（如果对象还存在）或 `nullptr`
- 常用场景：cache、observer 模式、打破循环引用

**【追问/扩展】**
- **`enable_shared_from_this`**：让一个对象从自身创建 `shared_ptr`，避免重复创建控制块
- **`shared_ptr` 的性能问题**：多线程频繁拷贝 `shared_ptr` 时，原子引用计数可能成为瓶颈（cache line bouncing）
- **`unique_ptr` 数组**：`std::unique_ptr<int[]> arr(new int[10])` 或 C++20 的 `make_unique_for_overwrite`
- **CUDA 中的智能指针使用**：`unique_ptr + CudaDeleter` 管理 device memory；`shared_ptr` 管理跨模块共享的 GPU 资源（如 cuBLAS handle）

---

【仔细】 ## 10.2 Move 语义和右值引用？std::move 的作用？

**【口述版】**
右值引用（`T&&`）可以绑定到即将销毁的临时对象；move 语义允许"偷取"临时对象的资源（如内存指针）而不是深拷贝；`std::move` 本身不做任何移动，只是把左值转换成右值引用（即 `static_cast<T&&>`），让编译器选择 move constructor/assignment。

**【详细版】**

**左值 vs 右值**：
```cpp
int x = 42;        // x 是左值（有名字，有地址）
int&& r = 42;      // 42 是右值（纯右值），r 是右值引用
// 但 r 本身是左值！（有名字）
```

**Move 构造函数**：
```cpp
class Buffer {
    float* data_;
    size_t size_;
public:
    // Copy constructor: 深拷贝，O(n)
    Buffer(const Buffer& other) : size_(other.size_) {
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }

    // Move constructor: 偷指针，O(1)
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;  // 源对象置空，防止 double free
        other.size_ = 0;
    }
};
```

**`std::move` 的本质**：
```cpp
// std::move 的实现（简化）
template<typename T>
constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
    return static_cast<std::remove_reference_t<T>&&>(t);
}
```
- **不做任何移动操作**，只做类型转换
- 效果：允许编译器选择 move 重载而不是 copy 重载

**应用场景**：
```cpp
std::vector<Buffer> buffers;
Buffer b(1024);
buffers.push_back(std::move(b));  // move 进 vector，避免拷贝
// b 现在处于 "valid but unspecified" 状态

// 函数返回：NRVO 优先，不需要 std::move
Buffer create() {
    Buffer b(1024);
    return b;  // NRVO 或隐式 move，不要写 std::move(b)
}
```

**完美转发（Perfect Forwarding）**：
```cpp
template<typename... Args>
auto make_buffer(Args&&... args) {
    return Buffer(std::forward<Args>(args)...);
}
// std::forward 保持参数的左值/右值属性
// 左值传入 → 左值传出；右值传入 → 右值传出
```

**【追问/扩展】**
- **`noexcept` 的重要性**：`vector` 扩容时只有 move constructor 是 `noexcept` 才会用 move（否则回退到 copy 以保证异常安全）
- **返回值优化（RVO/NRVO）**：编译器直接在调用者的栈上构造，连 move 都不需要；不要对 return 语句加 `std::move`，会抑制 NRVO
- **移动后的状态**：标准只保证对象处于 "valid but unspecified" 状态，通常实现为 "空" 状态
- **`std::move` vs `std::forward`**：`move` 无条件转为右值引用；`forward` 条件性保持原有类型（用于模板中的完美转发）

---

【仔细】 ## 10.3 虚函数和多态？虚函数表（vtable）的实现？

**【口述版】**
虚函数通过 vtable（虚函数表）实现运行时多态。每个含虚函数的类有一个 vtable（存放虚函数指针），每个对象有一个 vptr 指向类的 vtable。调用虚函数时通过 vptr 间接跳转到实际函数，多一次间接寻址的开销。

**【详细版】**

**基本用法**：
```cpp
class Shape {
public:
    virtual double area() const = 0;  // 纯虚函数
    virtual ~Shape() = default;        // 虚析构函数
};

class Circle : public Shape {
    double r_;
public:
    Circle(double r) : r_(r) {}
    double area() const override { return 3.14159 * r_ * r_; }
};

class Rect : public Shape {
    double w_, h_;
public:
    Rect(double w, double h) : w_(w), h_(h) {}
    double area() const override { return w_ * h_; }
};

void print_area(const Shape& s) {
    std::cout << s.area() << "\n";  // 运行时决定调用哪个 area()
}
```

**vtable 内存布局**：
```
Shape 的 vtable:
  [0] → Shape::area (纯虚, 调用会 abort)
  [1] → Shape::~Shape

Circle 的 vtable:
  [0] → Circle::area
  [1] → Circle::~Circle (调用 Shape::~Shape)

Circle 对象内存:
  +0: vptr → Circle 的 vtable
  +8: r_ (double)
```

**虚函数调用过程**：
```
s.area()
  → 读取 s 的 vptr（对象首部）
  → 从 vtable 中取 area 的函数指针（vtable[0]）
  → 通过函数指针间接调用
```

**性能开销**：
- 每个对象多一个 vptr（通常 8 字节）
- 每次虚函数调用多一次间接寻址（读 vptr + 读 vtable 条目 + 间接跳转）
- **无法 inline**：编译器看不到具体调用目标（除非能 devirtualize）
- 对于 hot loop 中的小函数，虚函数开销可能显著

**【追问/扩展】**
- **多重继承时的 vtable**：每个基类一个 vtable，对象中有多个 vptr，调用时需要 this 指针调整
- **`final` 关键字**：`class Derived final` 或 `void f() final`，编译器可以 devirtualize（消除虚函数调用）
- **`override` 关键字**：编译期检查是否真正覆盖了基类虚函数，防止拼写错误
- **CRTP 替代方案**：编译期多态，零运行时开销（见 10.14）

---

【仔细】 ## 10.4 C++ 内存模型？堆 / 栈 / 静态区？

**【口述版】**
C++ 程序内存分为：**栈**（自动存储，函数局部变量，LIFO，快速分配释放）、**堆**（动态分配 new/malloc，需手动释放或用智能指针）、**静态/全局区**（全局变量、static 变量，程序生命周期）、**代码段**（只读，存放指令）、**常量区**（字符串字面量等）。

**【详细版】**

**内存布局**（从高地址到低地址，典型 Linux）：
```
高地址
┌──────────────┐
│   Kernel Space│  (用户不可访问)
├──────────────┤
│     Stack     │  ← 向低地址增长
│               │  函数局部变量、参数、返回地址
│       ↓       │
│               │
│       ↑       │
│     Heap      │  ← 向高地址增长
│               │  new / malloc 分配
├──────────────┤
│     BSS       │  未初始化的全局/static 变量（零初始化）
├──────────────┤
│     Data      │  已初始化的全局/static 变量
├──────────────┤
│     Text      │  代码段（只读）
└──────────────┘
低地址
```

**各区域对比**：

| 区域 | 分配方式 | 生命周期 | 速度 | 大小限制 |
|---|---|---|---|---|
| **栈** | 自动（编译器管理） | 函数作用域 | 极快（移动 SP） | 默认 ~8MB（Linux） |
| **堆** | 手动（new/malloc） | 程序员控制 | 较慢（系统调用） | 受限于 virtual memory |
| **静态区** | 编译期分配 | 程序整个生命周期 | N/A | 受限于可执行文件大小 |

**栈的细节**：
```cpp
void foo() {
    int x = 10;          // 栈上分配，函数返回自动释放
    int arr[1024];       // 栈上数组（不要太大！）
    std::array<int, 4> a; // 栈上
}
// 栈帧：返回地址 | 上一帧指针 | 局部变量 | ...
```

**堆的细节**：
```cpp
auto p = new int(42);     // 堆分配
delete p;                  // 手动释放
auto v = std::make_unique<std::vector<int>>(1000);
// vector 对象在堆上，其内部 buffer 也在堆上
// unique_ptr 析构时自动 delete
```

**内存分配器**：
- `malloc` / `free`：C 标准库，底层调用 `brk` / `mmap`
- `new` / `delete`：C++ 运算符，内部通常调用 `malloc` + 构造函数
- **tcmalloc / jemalloc**：高性能内存分配器，减少锁竞争（PyTorch 推荐使用 `jemalloc`）
- **Memory pool**：预分配大块内存，减少系统调用（CUDA 的 `cudaMemPool`）

**【追问/扩展】**
- **栈溢出**：递归过深或栈上分配过大数组，用 `ulimit -s` 查看/修改栈大小
- **CUDA 中的内存**：device memory 对应 GPU 的 HBM（通过 `cudaMalloc`）；host pinned memory 通过 `cudaMallocHost`，在 DMA 时不需要额外拷贝
- **Placement new**：在已分配的内存上构造对象，`new (ptr) T(...)`，常用于内存池
- **C++ 内存模型（并发语义）**：C++11 定义了 memory model，规定了多线程下共享变量的可见性保证（见 10.9 原子操作和内存序）

---

【仔细】 ## 10.5 RAII 原则？

**【口述版】**
RAII（Resource Acquisition Is Initialization）把资源的获取和释放绑定到对象的构造和析构。构造时获取资源，析构时释放资源，利用 C++ 的确定性析构保证资源不泄漏（即使发生异常）。智能指针、`lock_guard`、`fstream` 都是 RAII 的典型应用。

**【详细版】**

**核心思想**：
```cpp
class CudaStream {
    cudaStream_t stream_;
public:
    CudaStream() { cudaStreamCreate(&stream_); }   // 构造时创建
    ~CudaStream() { cudaStreamDestroy(stream_); }  // 析构时销毁

    CudaStream(const CudaStream&) = delete;            // 禁止拷贝
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    cudaStream_t get() const { return stream_; }
};

void compute() {
    CudaStream s;               // 创建 stream
    launch_kernel(s.get());     // 使用
    // 即使这里抛异常，s 的析构也会被调用
}   // 离开作用域，自动销毁 stream
```

**为什么 RAII 比手动管理好**：
```cpp
// 手动管理：容易忘记释放，异常时泄漏
void bad() {
    cudaStream_t s;
    cudaStreamCreate(&s);
    do_work(s);           // 如果这里抛异常 → s 泄漏！
    cudaStreamDestroy(s); // 可能执行不到
}
```

**标准库中的 RAII**：
| 类 | 管理的资源 |
|---|---|
| `std::unique_ptr` / `shared_ptr` | 堆内存 |
| `std::lock_guard` / `unique_lock` | mutex 锁 |
| `std::fstream` | 文件句柄 |
| `std::jthread` (C++20) | 线程 |
| `std::scoped_lock` | 多个 mutex |

**CUDA 相关的 RAII 封装**：
```cpp
// Device memory RAII
class DeviceBuffer {
    void* ptr_ = nullptr;
    size_t size_ = 0;
public:
    explicit DeviceBuffer(size_t size) : size_(size) {
        cudaMalloc(&ptr_, size);
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;

    void* get() const { return ptr_; }
    size_t size() const { return size_; }
};
```

**【追问/扩展】**
- **Rule of 0/3/5**：如果需要自定义析构函数，通常也需要自定义拷贝构造/赋值（Rule of 3）或加上 move（Rule of 5）；最好用智能指针实现 Rule of 0（不需要自定义任何特殊成员函数）
- **异常安全**：RAII 是实现异常安全的基础。栈展开（stack unwinding）时所有局部对象的析构函数会被调用
- **`std::scoped_lock`**：C++17，同时锁多个 mutex，避免死锁（内部用 `std::lock`）
- **RAII 在 CUDA 中的实际应用**：PyTorch 的 `c10::cuda::CUDAGuard` 就是用 RAII 管理当前设备切换

---

【仔细】 ## 10.6 模板元编程？SFINAE？Concepts (C++20)？

**【口述版】**
模板元编程利用 C++ 模板在编译期做计算和类型推导。SFINAE（Substitution Failure Is Not An Error）是模板的核心规则：模板参数替换失败时不报错而是尝试下一个重载。C++20 的 Concepts 是对 SFINAE 的高级封装，用声明式语法约束模板参数，代码更清晰。

**【详细版】**

**模板元编程基础**：
```cpp
// 编译期计算阶乘
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};
static_assert(Factorial<5>::value == 120);
```

**SFINAE 示例**：
```cpp
// 只对浮点类型启用
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
my_sqrt(T x) {
    return std::sqrt(x);
}

// 替换 int 时 enable_if 条件不满足 → 替换失败 → 不报错
// my_sqrt(42); // 编译错误：没有匹配的重载（不是 SFINAE 错误）
my_sqrt(42.0);  // OK
```

**C++17 的 `if constexpr`**：
```cpp
template<typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>) {
        return val * 2;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(val);
    } else {
        static_assert(false, "unsupported type");
    }
}
```

**C++20 Concepts**：
```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept GpuBuffer = requires(T t) {
    { t.data() } -> std::convertible_to<void*>;
    { t.size() } -> std::convertible_to<size_t>;
    { t.device() } -> std::same_as<int>;
};

// 使用 concept 约束模板
template<Numeric T>
T add(T a, T b) { return a + b; }

// 或者用 requires 子句
template<typename T>
    requires GpuBuffer<T>
void launch_kernel(const T& buf) { /* ... */ }
```

**在 CUDA/AI 代码中的应用**：
```cpp
// CUTLASS 大量使用模板元编程
// 编译期选择 GEMM 的 tile size、数据类型、epilogue
using GemmOp = cutlass::gemm::device::Gemm<
    cutlass::half_t,                    // ElementA
    cutlass::layout::RowMajor,          // LayoutA
    cutlass::half_t,                    // ElementB
    cutlass::layout::ColumnMajor,       // LayoutB
    cutlass::half_t,                    // ElementC
    cutlass::layout::RowMajor,          // LayoutC
    float,                              // ElementAccumulator
    cutlass::arch::OpClassTensorOp,     // 使用 Tensor Core
    cutlass::arch::Sm80,                // 目标架构
    cutlass::gemm::GemmShape<128,256,64> // Tile shape
>;
```

**【追问/扩展】**
- **`std::void_t`（C++17）**：SFINAE 的常用工具，检测表达式是否合法
- **`constexpr` 函数 vs 模板元编程**：现代 C++ 倾向用 `constexpr` 函数替代模板递归（更可读）
- **Concepts 的错误信息**：比 SFINAE 好得多，编译器会直接说"不满足 concept XYZ"
- **编译时间问题**：重度模板元编程（如 CUTLASS）会显著增加编译时间

---

【仔细】 ## 10.7 std::vector 的内存布局和扩容策略？

**【口述版】**
`std::vector` 在堆上维护一段连续内存，有三个指针：begin（起始）、end（当前末尾）、end_of_storage（已分配空间末尾）。当 `size() == capacity()` 时插入元素触发扩容：分配 2x（或 1.5x）新内存 → 移动/拷贝旧元素 → 释放旧内存。扩容会导致所有迭代器和指针失效。

**【详细版】**

**内存布局**：
```
vector<int> v = {1, 2, 3};   // capacity 可能是 4

栈上的 vector 对象（3 个指针，24 字节）:
  _begin         → [1][2][3][?]  ← 堆上的连续内存
  _end           ────────↑   (指向最后一个元素的下一个位置)
  _end_of_storage────────────↑  (指向分配空间的末尾)

size()     = _end - _begin           = 3
capacity() = _end_of_storage - _begin = 4
```

**扩容流程**：
```cpp
v.push_back(4);  // size == capacity，触发扩容
// 1. 分配新内存：new_cap = old_cap * 2（GCC）或 * 1.5（MSVC）
// 2. 移动旧元素到新内存（如果 T 有 noexcept move，就 move；否则 copy）
// 3. 析构旧内存中的元素
// 4. 释放旧内存
// 5. 更新三个指针
```

**为什么是 2x 或 1.5x**：
- **2x（GCC libstdc++）**：均摊 `push_back` 复杂度 O(1)，但新分配的空间永远不会覆盖旧空间（无法复用之前释放的内存块）
- **1.5x（MSVC）**：几次扩容后，之前释放的小块内存之和可能超过当前需要的大小，可以复用（对内存碎片更友好）
- 数学上只要增长因子 > 1 就能保证均摊 O(1)

**性能注意事项**：
```cpp
// 好：预分配避免扩容
std::vector<float> v;
v.reserve(1000000);  // 一次分配，避免多次扩容

// 坏：频繁小量 push_back
std::vector<float> v;
for (int i = 0; i < 1000000; i++)
    v.push_back(i);  // 多次扩容，拷贝大量数据

// emplace_back vs push_back
v.emplace_back(args...);  // 原地构造，避免临时对象
v.push_back(T(args...));  // 先构造临时对象，再 move 进 vector
```

**与 CUDA 的关系**：
- `std::vector<float>` 在 host 上，是 pinned memory 的常见来源：
```cpp
std::vector<float> h_data(N);
// 拷贝到 device
cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
```
- 但 `std::vector` 的内存不是 pinned 的，H2D 拷贝时 CUDA 会先拷贝到内部 staging buffer
- 需要 pinned memory 时用 `cudaMallocHost` 或自定义 allocator

**【追问/扩展】**
- **`shrink_to_fit()`**：请求释放多余容量（非强制，实现可以忽略）
- **`reserve` vs `resize`**：`reserve` 只分配内存不构造元素；`resize` 构造元素
- **迭代器失效**：扩容后所有迭代器、指针、引用都失效；`insert` / `erase` 也可能导致失效
- **`std::vector<bool>` 的坑**：特化为 bitset，每元素 1 bit，`operator[]` 返回代理对象而非 `bool&`，不是真正的容器

---

【仔细】 ## 10.8 多线程编程？std::thread / mutex / condition_variable？

**【口述版】**
`std::thread` 创建线程，`std::mutex` + `lock_guard` 保护共享数据，`std::condition_variable` 实现线程间的等待-通知机制。C++ 多线程在 AI 系统中常用于数据加载（DataLoader worker）、异步 GPU 操作、以及推理服务的请求处理。

**【详细版】**

**基本用法**：
```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// 线程安全的生产者-消费者队列
template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;

public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push(std::move(value));
        }  // 先解锁再通知（减少不必要的阻塞）
        cv_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool try_pop(T& value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
            return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
};
```

**`std::thread`**：
```cpp
void worker(int id) { /* ... */ }

std::thread t1(worker, 1);
std::thread t2(worker, 2);

t1.join();   // 等待 t1 完成
t2.detach(); // 分离 t2（在后台运行，不能 join）

// C++20 jthread：RAII 管理，析构时自动 join + 支持 stop_token
std::jthread jt([](std::stop_token st) {
    while (!st.stop_requested()) {
        /* work */
    }
});
```

**锁的种类**：
| 锁 | 特点 |
|---|---|
| `std::mutex` | 基础互斥锁 |
| `std::recursive_mutex` | 同一线程可重复加锁 |
| `std::shared_mutex` (C++17) | 读写锁：多读单写 |
| `std::lock_guard` | RAII 锁，不可中途解锁 |
| `std::unique_lock` | RAII 锁，可中途 unlock/lock，可配合 cv |
| `std::scoped_lock` (C++17) | 同时锁多个 mutex，防死锁 |

**Condition Variable 的注意事项**：
- 必须和 `unique_lock` 配合（`lock_guard` 不行，因为 cv.wait 需要临时 unlock）
- **Spurious wakeup**：`wait` 可能假唤醒，必须用谓词版本 `wait(lock, predicate)` 或在循环中检查条件
- `notify_one` 唤醒一个等待线程；`notify_all` 唤醒所有

**【追问/扩展】**
- **死锁预防**：`std::scoped_lock(m1, m2)` 内部用 `std::lock` 算法避免死锁；或者约定加锁顺序
- **线程池**：生产中不直接创建 `std::thread`，而是用线程池复用线程（减少创建销毁开销）
- **`std::async` / `std::future`**：更高层的异步接口，返回 `future` 获取结果
- **在 AI 系统中的应用**：PyTorch DataLoader 的多 worker 用多进程（Python GIL 限制）而非多线程；但 C++ inference server（如 TensorRT-LLM）用多线程处理并发请求

---

【仔细】 ## 10.9 原子操作和内存序（memory_order）？

**【口述版】**
`std::atomic<T>` 提供无锁的原子操作，保证读-修改-写的原子性。内存序（`memory_order`）控制原子操作前后的内存可见性：`relaxed`（只保证原子性，不保证顺序）、`acquire/release`（建立 happens-before 关系，同步数据）、`seq_cst`（最强，所有线程看到相同的操作顺序）。

**【详细版】**

**`std::atomic` 基本操作**：
```cpp
std::atomic<int> counter{0};

// 原子 load / store
int val = counter.load(std::memory_order_relaxed);
counter.store(42, std::memory_order_relaxed);

// 原子 read-modify-write
counter.fetch_add(1, std::memory_order_relaxed);  // counter++
counter.fetch_sub(1);  // counter--

// CAS (Compare-And-Swap)
int expected = 0;
bool success = counter.compare_exchange_strong(
    expected, 1, std::memory_order_acq_rel);
// 如果 counter == expected (0)，则设为 1 并返回 true
// 否则 expected 被更新为 counter 的当前值，返回 false
```

**六种内存序**：

| 内存序 | 含义 | 开销 |
|---|---|---|
| `relaxed` | 只保证原子性，不限制重排 | 最低 |
| `consume` | 数据依赖的 acquire（几乎没人用） | - |
| `acquire` | 此操作后的读写不能重排到此操作前 | 中等 |
| `release` | 此操作前的读写不能重排到此操作后 | 中等 |
| `acq_rel` | acquire + release | 中等 |
| `seq_cst` | 全序一致（默认，最安全） | 最高 |

**Acquire-Release 语义（最常用的同步模式）**：
```cpp
std::atomic<bool> ready{false};
int data = 0;

// Thread 1 (Producer)
data = 42;                                    // (a)
ready.store(true, std::memory_order_release); // (b) release

// Thread 2 (Consumer)
while (!ready.load(std::memory_order_acquire)) {} // (c) acquire
assert(data == 42);  // 保证成功！
// acquire 与 release 配对：(b) happens-before (c)
// 因此 (a) happens-before assert
```

**在 CUDA 中的对应**：
- CUDA 的 `atomicAdd`、`atomicCAS` 等是 GPU 端的原子操作
- CUDA 的 `__threadfence()`、`__threadfence_block()`、`__threadfence_system()` 对应不同范围的 memory fence
- CUDA 内存模型（自 CUDA 12 起）逐渐对齐 C++ 内存模型

**【追问/扩展】**
- **`seq_cst` 的开销**：x86 上 store 需要 `MFENCE` 或 `LOCK XCHG`，ARM 上需要 full barrier，比 relaxed 慢很多
- **Lock-free vs Wait-free**：lock-free 保证系统整体前进；wait-free 保证每个线程都在有限步内完成
- **False sharing**：两个不相关的 `atomic` 变量在同一 cache line 上，互相拖慢（见 10.17）
- **在 CUDA atomicAdd 中**：GPU 的 atomic 粒度可以是 block-level（`__threadfence_block`）或 device-level（`__threadfence`），选择合适粒度可以减少开销

---

【仔细】 ## 10.10 Lambda 表达式的实现？capture list？

**【口述版】**
Lambda 在编译器内部被转换成一个匿名类（functor），重载了 `operator()`。Capture list 决定怎么捕获外部变量：`[=]` 按值拷贝、`[&]` 按引用、`[x]` 显式按值、`[&x]` 显式按引用。按值捕获时值在 lambda 创建时确定，按引用捕获时引用可能悬空。

**【详细版】**

**编译器变换**：
```cpp
int x = 10;
auto f = [x](int y) { return x + y; };
// 等价于：
struct __lambda_1 {
    int x;  // 按值捕获的成员
    int operator()(int y) const { return x + y; }
};
auto f = __lambda_1{x};
```

**Capture 方式**：
```cpp
int a = 1, b = 2, c = 3;

auto f1 = [a, &b]() { /* a 按值, b 按引用 */ };
auto f2 = [=]()     { /* 所有使用的变量按值 */ };
auto f3 = [&]()     { /* 所有使用的变量按引用 */ };
auto f4 = [=, &b]() { /* 默认按值, b 按引用 */ };
auto f5 = [this]()  { /* 捕获 this 指针 */ };
auto f6 = [*this]() { /* 捕获 this 对象的拷贝 (C++17) */ };

// C++14: init capture（初始化捕获）
auto f7 = [x = std::move(some_unique_ptr)]() { /* x 是 move 进来的 */ };
auto f8 = [v = std::vector<int>{1,2,3}]() { /* v 在 lambda 内 */ };
```

**Mutable lambda**：
```cpp
int x = 0;
auto f = [x]() mutable { return ++x; };
// 没有 mutable 时 operator() 是 const，不能修改按值捕获的变量
f(); // 返回 1
f(); // 返回 2（lambda 内部的 x 副本在递增）
```

**`std::function` 的开销**：
```cpp
// Lambda 本身是零开销（编译器直接 inline）
auto f = [](int x) { return x * 2; };

// std::function 有类型擦除开销（堆分配 + 虚调用）
std::function<int(int)> g = f;
// 如果 lambda 捕获的数据 > SBO 阈值（通常 16-32 字节），会堆分配
```

**【追问/扩展】**
- **悬空引用陷阱**：按引用捕获局部变量后，如果 lambda 生命周期超过变量作用域，引用悬空导致 UB
- **泛型 lambda（C++14）**：`auto f = [](auto x) { return x; }` — 内部生成模板 `operator()`
- **Lambda 大小**：无捕获的 lambda 大小为 1 字节；每按值捕获一个变量就增加该变量的大小
- **无捕获 lambda 可以转函数指针**：`void (*fp)(int) = [](int x) { printf("%d", x); };`
- **在 CUDA 中**：`__device__ lambda` 在 CUDA kernel 中使用（需要编译器支持），Thrust/CUB 等库大量使用 lambda 作为用户自定义操作

---

【仔细】 ## 10.11 C++ 异常处理机制？CUDA 中的错误处理？

**【口述版】**
C++ 用 `try-catch-throw` 机制处理异常，运行时代价（栈展开 + RTTI）主要在异常抛出时产生，正常路径几乎零开销。CUDA 不支持 device 端异常，用返回码（`cudaError_t`）+ `cudaGetLastError()` 处理错误；实践中用宏封装错误检查。

**【详细版】**

**C++ 异常机制**：
```cpp
void might_fail() {
    if (error_condition)
        throw std::runtime_error("something went wrong");
}

try {
    might_fail();
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
} catch (...) {
    std::cerr << "Unknown exception" << std::endl;
}
```

**实现机制（零开销异常模型）**：
- 编译器生成 **异常表（exception table）**，记录每个 try block 的范围和对应的 catch handler
- **正常路径零开销**：不检查返回值，不做额外跳转
- **抛出异常时**：搜索异常表 → 栈展开（调用析构函数）→ 找到匹配的 catch handler → 跳转
- 异常抛出的开销很大（微秒级），但正常路径几乎免费

**CUDA 错误处理**：
```cpp
// CUDA API 返回错误码
#define CUDA_CHECK(call) do {                               \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                __FILE__, __LINE__,                         \
                cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// 使用
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));

// Kernel launch 后的检查
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());     // 检查 launch 参数错误
CUDA_CHECK(cudaDeviceSynchronize()); // 检查执行时错误
```

**CUDA 错误的特殊性**：
- **异步性**：kernel launch 是异步的，错误可能在 launch 之后才被检测到
- **Sticky error**：某些错误（如 illegal memory access）是 "sticky" 的，一旦发生，之后所有 CUDA API 都返回该错误，只能重置 device
- **`cudaGetLastError()` vs `cudaPeekAtLastError()`**：前者读取并清除错误，后者只读取不清除

**PyTorch 的做法**：
```cpp
// PyTorch 用 C10_CUDA_CHECK 宏，抛出 C++ 异常
C10_CUDA_CHECK(cudaMalloc(&ptr, size));
// 内部：if (err != cudaSuccess) throw c10::CUDAError(...)
```

**【追问/扩展】**
- **`noexcept` 的作用**：声明函数不会抛异常；如果还是抛了，直接 `std::terminate`。编译器可以据此做优化（如 move 操作）
- **异常 vs 错误码 vs `std::expected` (C++23)**：高性能代码（如 CUDA runtime）用错误码；业务逻辑用异常；C++23 的 `expected` 提供了类型安全的错误处理
- **`-fno-exceptions` 编译选项**：关闭异常支持，所有 `throw` 变成 `abort`；一些嵌入式/游戏引擎项目使用
- **CUDA 的 `assert()` 在 device 端**：可以用 `assert()` 在 kernel 内打断，但只用于调试（触发后整个 context 不可用）

---

【仔细】 ## 10.12 虚析构函数的作用？

**【口述版】**
当通过基类指针 `delete` 派生类对象时，如果基类析构函数不是虚的，只会调用基类的析构函数而不会调用派生类的析构函数，导致资源泄漏（未定义行为）。声明基类析构函数为 `virtual` 确保析构时走 vtable 调用正确的派生类析构函数。

**【详细版】**

**问题演示**：
```cpp
class Base {
    int* data_;
public:
    Base() : data_(new int[100]) {}
    ~Base() { delete[] data_; }  // 非虚析构
};

class Derived : public Base {
    float* gpu_data_;
public:
    Derived() { cudaMalloc(&gpu_data_, 1024); }
    ~Derived() { cudaFree(gpu_data_); }  // 永远不会被调用！
};

Base* p = new Derived();
delete p;  // 只调用 Base::~Base()，Derived::~Derived() 没被调用
           // gpu_data_ 泄漏！而且这是 UB
```

**修复**：
```cpp
class Base {
    int* data_;
public:
    Base() : data_(new int[100]) {}
    virtual ~Base() { delete[] data_; }  // 虚析构
};
// 现在 delete p 会先调用 Derived::~Derived()，再调用 Base::~Base()
```

**规则**：
- 只要一个类**可能被继承**且**可能通过基类指针删除**，析构函数就应该是 `virtual`
- 如果类不打算被继承，用 `final` 标记
- 如果不需要多态（不通过基类指针使用），不需要虚析构（避免 vtable 开销）

**析构顺序**：
```
delete derived_ptr;
  → Derived::~Derived()   // 先析构派生类成员
    → Base::~Base()       // 再析构基类成员
      // 成员变量按声明逆序析构
```

**【追问/扩展】**
- **纯虚析构函数**：可以声明 `virtual ~Base() = 0;` 但**必须提供定义**（因为派生类析构会调用基类析构）
- **`protected` 非虚析构**：另一种防止通过基类指针删除的方式——把析构函数设为 `protected`，外部无法 `delete`
- **`shared_ptr` 的类型擦除**：`shared_ptr<Base>` 即使基类析构非虚，也能正确调用派生类析构（因为 deleter 在创建时就绑定了具体类型）。但 `unique_ptr<Base>` 不行
- **性能考虑**：虚析构意味着类有 vtable，每个对象多一个 vptr（8 字节），如果有大量小对象可能有影响

---

【仔细】 ## 10.13 const / constexpr / consteval 的区别？

**【口述版】**
`const` 表示运行时常量（值不可修改但运行时才确定）；`constexpr` 表示**可以**在编译期求值（也可以在运行时）；`consteval`（C++20）表示**必须**在编译期求值（否则编译错误）。

**【详细版】**

**`const`**：
```cpp
const int x = 42;           // 编译期常量（编译器可能优化为立即数）
const int y = get_value();   // 运行时常量（值运行时确定，之后不可改）

void foo(const std::vector<int>& v) {
    // v 的内容不能通过此引用修改
    // 但如果有其他非 const 引用，对象本身可以变
}

class MyClass {
    int data_;
public:
    int get() const { return data_; }  // const 成员函数，不修改对象
    // 内部 this 类型是 const MyClass*
};
```

**`constexpr`（C++11/14/17/20 逐步增强）**：
```cpp
constexpr int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

constexpr int f5 = factorial(5);     // 编译期求值
int n;
std::cin >> n;
int fn = factorial(n);               // 运行时求值（也合法）

// constexpr 变量：必须编译期初始化
constexpr int SIZE = 1024;
constexpr auto PI = 3.14159265358979;

// C++17: constexpr if
template<typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>) {
        return val * 2;
    } else {
        return val + 0.5;
    }
}

// C++20: constexpr 支持 new/delete、虚函数、try-catch 等
constexpr std::vector<int> make_vec() {
    std::vector<int> v = {1, 2, 3};
    v.push_back(4);
    return v;
}
```

**`consteval`（C++20）**：
```cpp
consteval int must_be_compile_time(int x) {
    return x * x;
}

constexpr int a = must_be_compile_time(5);  // OK
int b = must_be_compile_time(5);            // OK（参数是常量）

int n = 5;
// int c = must_be_compile_time(n);  // 编译错误！n 不是编译期常量
```

**`constinit`（C++20）**：
```cpp
constinit int global = 42;  // 保证静态变量编译期初始化
                              // 解决 static initialization order fiasco
// constinit 不意味着 const，之后可以修改 global
```

**对比总结**：

| 特性 | `const` | `constexpr` | `consteval` | `constinit` |
|---|---|---|---|---|
| 编译期求值 | 不保证 | 可以 | 必须 | 初始化时必须 |
| 运行时可变 | 不可变 | 不可变（变量）/ 可运行时调用（函数） | N/A | 可变 |
| 引入版本 | C++ | C++11 | C++20 | C++20 |

**【追问/扩展】**
- **`mutable` 关键字**：允许在 `const` 成员函数中修改特定成员（如 cache、mutex）
- **`const_cast`**：移除 const 限定，对本身不是 const 的对象合法，对真正 const 的对象 UB
- **`constexpr` 函数的限制**：C++11 很严格（只能有 return 语句）；C++14 放宽（允许循环、局部变量）；C++20 几乎无限制
- **在 CUDA 中**：`__device__ constexpr` 函数可以在编译期和 device 端运行时调用

---

【仔细】 ## 10.14 编译期多态 vs 运行时多态？CRTP 模式？

**【口述版】**
运行时多态通过虚函数 + vtable 实现，有间接调用开销；编译期多态通过模板实现，零运行时开销但会增加编译时间和二进制大小。CRTP（Curiously Recurring Template Pattern）是编译期多态的经典模式：`class Derived : public Base<Derived>`，基类通过 `static_cast<Derived*>(this)` 调用派生类方法。

**【详细版】**

**运行时多态（virtual）**：
```cpp
class Kernel {
public:
    virtual void launch(float* data, int n) = 0;
    virtual ~Kernel() = default;
};
class AddKernel : public Kernel {
    void launch(float* data, int n) override { /* ... */ }
};

void run(Kernel& k, float* data, int n) {
    k.launch(data, n);  // 虚调用，运行时查表
}
```

**编译期多态（CRTP）**：
```cpp
template<typename Derived>
class KernelBase {
public:
    void launch(float* data, int n) {
        // 调用派生类的实现（编译期确定）
        static_cast<Derived*>(this)->launch_impl(data, n);
    }
};

class AddKernel : public KernelBase<AddKernel> {
public:
    void launch_impl(float* data, int n) {
        // 具体实现
    }
};

template<typename K>
void run(KernelBase<K>& k, float* data, int n) {
    k.launch(data, n);  // 编译期确定，可以 inline
}
```

**CRTP 的应用场景**：

**1. Mixin 模式（添加功能）**：
```cpp
template<typename Derived>
class Printable {
public:
    void print() const {
        auto& d = static_cast<const Derived&>(*this);
        std::cout << d.to_string() << std::endl;
    }
};

class MyClass : public Printable<MyClass> {
public:
    std::string to_string() const { return "MyClass"; }
};
```

**2. 在 CUTLASS 中**（大量使用 CRTP）：
```cpp
template<typename Derived>
class GemmBase {
public:
    void run() {
        auto& impl = static_cast<Derived&>(*this);
        impl.prologue();
        impl.mainloop();
        impl.epilogue();
    }
};
```

**对比**：

| 维度 | 运行时多态（virtual） | 编译期多态（CRTP/template） |
|---|---|---|
| 决定时机 | 运行时 | 编译时 |
| 性能 | 间接调用，无法 inline | 直接调用，可 inline |
| 灵活性 | 可以存在异构容器中 | 不同 Derived 是不同类型 |
| 二进制大小 | 较小 | 每个实例化都生成代码 |
| 编译时间 | 快 | 慢（模板实例化） |
| 调试 | 容易 | 模板错误信息晦涩 |

**C++20 的 Concepts 让编译期多态更清晰**：
```cpp
template<typename T>
concept KernelLike = requires(T t, float* data, int n) {
    { t.launch(data, n) } -> std::same_as<void>;
};

void run(KernelLike auto& k, float* data, int n) {
    k.launch(data, n);
}
```

**【追问/扩展】**
- **虚函数的 devirtualization**：编译器在能确定具体类型时（如 `final` 类、局部变量）会自动消除虚调用
- **Type erasure（类型擦除）**：`std::function`、`std::any` 内部结合了模板和虚函数，提供运行时多态但隐藏了类型
- **CRTP 的陷阱**：`Derived` 不能是 `final`；基类不能直接用 `Derived` 的成员（可能还未定义）；析构函数问题（基类析构非虚，不能通过基类指针 delete）
- **为什么 CUTLASS 选择编译期多态**：GPU kernel 模板参数在编译期确定（tile size、数据类型等），不需要运行时分发；编译期确定能让编译器做最大化优化

---

【随意】 ## 10.15 std::optional / std::variant / std::any（C++17）？

**【口述版】**
`std::optional<T>` 表示"可能有值也可能为空"（替代裸指针或 sentinel value）；`std::variant<T1,T2,...>` 是类型安全的 union（同一时刻持有其中一种类型的值）；`std::any` 可以持有任意类型的值（类型擦除，有堆分配开销）。三者都是值语义的。

**【详细版】**

**`std::optional<T>`**：
```cpp
std::optional<int> find_index(const std::vector<int>& v, int target) {
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == target) return i;
    }
    return std::nullopt;  // 没找到
}

auto result = find_index(v, 42);
if (result.has_value()) {
    std::cout << "Found at index " << result.value() << "\n";
    // 或者 *result
}
// value_or 提供默认值
int idx = result.value_or(-1);
```
- 内部：`aligned_storage<T>` + bool 标志，无堆分配
- 大小：`sizeof(T) + 对齐padding + 1字节标志`
- 比返回 `-1` 或 `nullptr` 更安全、更表意

**`std::variant<Types...>`**：
```cpp
using Value = std::variant<int, float, std::string>;
Value v = 42;          // 持有 int
v = 3.14f;             // 持有 float
v = "hello"s;          // 持有 string

// 访问
std::visit([](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, int>)
        std::cout << "int: " << arg << "\n";
    else if constexpr (std::is_same_v<T, float>)
        std::cout << "float: " << arg << "\n";
    else
        std::cout << "string: " << arg << "\n";
}, v);

// 安全访问
auto* p = std::get_if<int>(&v);  // 返回指针，类型不匹配返回 nullptr
int i = std::get<int>(v);        // 类型不匹配抛 bad_variant_access
```
- 大小：`max(sizeof(Types...))` + index（通常 1-4 字节）
- 无堆分配，类型安全的 union

**`std::any`**：
```cpp
std::any a = 42;
a = std::string("hello");
a = 3.14;

// 访问
try {
    auto s = std::any_cast<std::string>(a);  // 抛 bad_any_cast
} catch (const std::bad_any_cast& e) { /* ... */ }

auto* p = std::any_cast<double>(&a);  // 返回指针，安全
```
- 内部用 SBO（Small Buffer Optimization）：小对象在栈上，大对象堆分配
- 类型信息用 `typeid` 存储，有 RTTI 开销

**对比**：

| 特性 | `optional` | `variant` | `any` |
|---|---|---|---|
| 可能的类型 | 1 种 + 空 | 编译期列举 | 任意 |
| 堆分配 | 无 | 无 | 可能（SBO） |
| 类型安全 | 是 | 是 | 运行时检查 |
| 使用场景 | 可能无值 | 有限的类型集合 | 完全动态 |

**【追问/扩展】**
- **`std::expected<T, E>`（C++23）**：比 `optional` 更好，可以携带错误信息（类似 Rust 的 `Result`）
- **`variant` 的 `valueless_by_exception`**：如果赋值时构造函数抛异常，variant 可能进入无值状态（极少发生但要注意）
- **在 AI 代码中的应用**：`optional` 用于可选参数（如 `std::optional<int> max_seq_len`）；`variant` 用于表示不同数据类型的 tensor（`variant<float, half, int8_t>`）
- **性能**：`optional` 和 `variant` 因为无堆分配，在 hot path 上比 `any` 和 `shared_ptr` 高效得多

---

【随意】 ## 10.16 内存对齐（alignment）？alignas / alignof？

**【口述版】**
内存对齐要求变量的地址是其对齐值（通常等于其大小）的倍数。对齐是为了匹配 CPU 的内存访问粒度——不对齐的访问在某些架构上直接崩溃，在 x86 上性能下降。`alignof` 查询类型的对齐要求，`alignas` 指定对齐值。

**【详细版】**

**基本规则**：
```cpp
// 基本类型的自然对齐
alignof(char)   == 1
alignof(short)  == 2
alignof(int)    == 4
alignof(double) == 8
alignof(void*)  == 8  // 64-bit 系统

// 结构体对齐 = 最大成员的对齐
struct S {
    char a;   // offset 0, size 1
              // 3 bytes padding
    int b;    // offset 4, size 4
    char c;   // offset 8, size 1
              // 3 bytes padding（尾部对齐到 4）
};
// sizeof(S) == 12, alignof(S) == 4
```

**`alignas` 指定对齐**：
```cpp
// 对齐到 cache line（64 字节）
struct alignas(64) CacheAligned {
    int data[16];  // 64 bytes
};
// sizeof(CacheAligned) == 64
// 分配地址一定是 64 的倍数

// 变量级别对齐
alignas(256) float buffer[1024];
// buffer 的地址是 256 的倍数，适合 SIMD 访问
```

**在 CUDA 中的重要性**：
```cpp
// GPU 全局内存访问：对齐到 16 字节可以启用向量化加载
struct alignas(16) Float4 {
    float x, y, z, w;
};

// Shared memory 对齐影响 bank conflict
__shared__ alignas(128) float smem[1024];

// cudaMalloc 返回的地址已经是 256 字节对齐的
// 但 host 内存可能需要手动对齐
void* ptr;
cudaMallocHost(&ptr, size);  // pinned memory，已对齐
```

**动态对齐分配**：
```cpp
// C++17 aligned new
auto p = new (std::align_val_t(64)) MyStruct;
// 或者
auto p = static_cast<float*>(std::aligned_alloc(64, size));
// 注意：size 必须是 64 的整数倍（aligned_alloc 的要求）
```

**【追问/扩展】**
- **`#pragma pack`**：强制减小对齐值（如 `#pragma pack(1)`），用于网络协议或文件格式，但会降低访问性能
- **过对齐（over-aligned）**：对齐值超过 `alignof(std::max_align_t)`（通常 16），C++17 前的 `new` 不保证支持，C++17 起自动支持
- **`std::aligned_storage` / `std::aligned_union`**（C++17 废弃）→ 使用 `alignas` + `char[]` 替代
- **SIMD 对齐**：AVX2 需要 32 字节对齐，AVX-512 需要 64 字节对齐；不对齐的 load 指令（`_mm256_loadu_ps`）比对齐版本慢

---

【仔细】 ## 10.17 Cache line 和 false sharing？

**【口述版】**
CPU cache 以 cache line（通常 64 字节）为单位加载和驱逐数据。False sharing 是指两个线程访问不同变量但这些变量在同一 cache line 上，导致 cache line 在核心间频繁 invalidate 和传输，严重降低多线程性能。解决方法是 padding 或使用 `alignas(64)` 让不同线程的数据在不同 cache line 上。

**【详细版】**

**Cache line 基础**：
```
内存地址：... [0x100-0x13F] [0x140-0x17F] [0x180-0x1BF] ...
                cache line 1   cache line 2   cache line 3
```
- 每次 cache miss 从内存加载一整个 cache line（64B on x86, 128B on ARM）
- 即使只需要 1 个 int（4B），也会加载 64B
- 空间局部性：相邻数据很可能会被访问，预加载是合理的

**False sharing 问题**：
```cpp
// 坏的设计：两个线程的计数器在同一 cache line
struct Counters {
    int counter_a;  // thread A 频繁修改
    int counter_b;  // thread B 频繁修改
};  // sizeof == 8, 两个 counter 在同一 cache line

Counters c;
// Thread A: c.counter_a++ (独占 cache line → invalidate B 的 cache)
// Thread B: c.counter_b++ (独占 cache line → invalidate A 的 cache)
// 不断 ping-pong，性能极差
```

**修复方法**：
```cpp
// 方法 1：padding
struct Counters {
    alignas(64) int counter_a;  // 独占一个 cache line
    alignas(64) int counter_b;  // 独占另一个 cache line
};

// 方法 2：C++17 hardware_destructive_interference_size
struct Counters {
    alignas(std::hardware_destructive_interference_size) int counter_a;
    alignas(std::hardware_destructive_interference_size) int counter_b;
};

// 方法 3：手动 padding
struct PaddedCounter {
    int value;
    char padding[60];  // 填充到 64 字节
};
```

**性能影响（实际测量）**：
```
False sharing：       ~100M ops/sec (2 threads)
Fixed (aligned):      ~2000M ops/sec (2 threads)
性能差异可达 10-50x！
```

**在 GPU 上的类比**：
- GPU 没有传统意义的 false sharing（不同 SM 的 L1 是独立的）
- 但有类似问题：
  - **Shared memory bank conflict**：多个线程访问同一 bank
  - **L2 cache thrashing**：不同 SM 的 warp 争用同一 L2 cache line
  - **Atomic 争用**：多个 warp 对同一地址做 atomicAdd

**【追问/扩展】**
- **`hardware_constructive_interference_size`**：建议在同一 cache line 中放置的最大大小（用于提高局部性，如把相关字段放在一起）
- **NUMA 和 false sharing**：跨 NUMA 节点的 false sharing 更严重（cache coherence 需要跨 socket 通信）
- **工具检测**：`perf c2c`（Linux）可以检测 false sharing；Intel VTune 也有相关分析
- **实际案例**：线程池中的任务队列、全局计数器、统计信息收集 — 都是 false sharing 高发区

---

【随意】 ## 10.18 C++ 与 Python 的交互？pybind11？

**【口述版】**
pybind11 是最流行的 C++/Python 绑定库，用 C++ 模板元编程自动处理类型转换和引用计数。只需简单的宏和绑定代码就能把 C++ 函数/类暴露给 Python 使用。PyTorch 的 C++ extension 机制底层就用 pybind11；另外还有 nanobind（pybind11 作者的轻量级后续项目）和 Python C API。

**【详细版】**

**pybind11 基本用法**：
```cpp
// my_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

float add(float a, float b) { return a + b; }

class MyKernel {
    int device_id_;
public:
    MyKernel(int device_id) : device_id_(device_id) {}
    void launch(py::array_t<float> input) {
        auto buf = input.request();
        float* ptr = static_cast<float*>(buf.ptr);
        int n = buf.size;
        // 调用 CUDA kernel...
    }
};

PYBIND11_MODULE(my_module, m) {
    m.doc() = "My CUDA module";

    m.def("add", &add, "Add two numbers",
          py::arg("a"), py::arg("b"));

    py::class_<MyKernel>(m, "MyKernel")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("launch", &MyKernel::launch);
}
```

**与 PyTorch 的集成**：
```cpp
// PyTorch C++ extension
#include <torch/extension.h>

torch::Tensor my_cuda_op(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    auto output = torch::empty_like(input);
    // 调用 CUDA kernel
    my_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_cuda_op", &my_cuda_op, "My CUDA operation");
}
```

**编译方式**：
```python
# setup.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension('my_module', [
            'my_module.cpp',
            'my_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)

# 或者 JIT 编译（开发调试方便）
from torch.utils.cpp_extension import load
my_module = load(name='my_module',
                 sources=['my_module.cpp', 'my_kernel.cu'])
```

**pybind11 的关键特性**：
- **自动类型转换**：Python int/float/str ↔ C++ int/float/std::string
- **NumPy 支持**：`py::array_t<T>` 零拷贝访问 numpy 数组
- **GIL 管理**：`py::gil_scoped_release` 释放 GIL 让 C++ 代码并行运行
- **引用计数整合**：C++ 对象的生命周期与 Python 的引用计数绑定
- **异常转换**：C++ 异常自动转为 Python 异常

**GIL 问题**：
```cpp
void heavy_compute(py::array_t<float> data) {
    py::gil_scoped_release release;  // 释放 GIL
    // 执行耗时计算（不访问 Python 对象）
    // 其他 Python 线程可以并行运行
}
```

**【追问/扩展】**
- **nanobind**：pybind11 作者的新项目，更轻量、编译更快、二进制更小；PyTorch 正在逐步迁移
- **ctypes / cffi**：不需要编译绑定代码，但类型安全差、功能少
- **Cython**：另一种方案，用 Python-like 语法写 C extension
- **性能注意**：Python → C++ 调用本身有开销（~100ns），如果调用太频繁（如每个 element）不如 batch 操作
- **torch.autograd.Function**：在 C++ extension 中实现自定义的 forward/backward，需要继承 `torch::autograd::Function`

---

【随意】 ## 10.19 常见的内存问题和调试工具？（valgrind / AddressSanitizer）

**【口述版】**
常见内存问题包括：内存泄漏、use-after-free、buffer overflow、double free、未初始化读取。调试工具主要有：Valgrind（Memcheck，运行时检查，慢 10-50x）、AddressSanitizer（ASan，编译期插桩，慢 2x，检测范围广）、CUDA 的 compute-sanitizer（检查 GPU 内存越界和 race condition）。

**【详细版】**

**常见内存问题**：

| 问题 | 说明 | 后果 |
|---|---|---|
| **Memory leak** | 分配后未释放 | 内存耗尽 |
| **Use-after-free** | 释放后还在使用 | 随机崩溃、数据损坏 |
| **Buffer overflow** | 越界读/写 | 安全漏洞、数据损坏 |
| **Double free** | 重复释放 | 崩溃、堆损坏 |
| **Uninitialized read** | 读取未初始化内存 | 不确定行为 |
| **Stack overflow** | 栈空间耗尽 | 段错误 |

**Valgrind (Memcheck)**：
```bash
# 编译时加 -g（保留调试信息）
g++ -g -o my_app my_app.cpp

# 运行 valgrind
valgrind --leak-check=full --show-reachable=yes ./my_app
```
- 优点：不需要重新编译（二进制级别检测）、检测精确
- 缺点：慢 10-50x、内存使用增加 2-3x
- 检测能力：内存泄漏、越界读写、未初始化读取、double free

**AddressSanitizer (ASan)**：
```bash
# 编译时启用
g++ -fsanitize=address -fno-omit-frame-pointer -g -o my_app my_app.cpp

# 直接运行，出错时输出详细报告
./my_app
```
- 优点：比 Valgrind 快得多（~2x slowdown）、错误报告详细
- 缺点：需要重新编译、增加内存使用（~3x）
- 检测能力：heap buffer overflow、stack buffer overflow、use-after-free、double free、memory leak

**其他 Sanitizer**：
```bash
# ThreadSanitizer (TSan): 检测 data race
g++ -fsanitize=thread -g -o my_app my_app.cpp

# MemorySanitizer (MSan): 检测未初始化内存读取
g++ -fsanitize=memory -g -o my_app my_app.cpp

# UndefinedBehaviorSanitizer (UBSan): 检测未定义行为
g++ -fsanitize=undefined -g -o my_app my_app.cpp
```

**CUDA 内存调试**：
```bash
# compute-sanitizer（替代已废弃的 cuda-memcheck）
compute-sanitizer --tool memcheck ./my_cuda_app
# 检测：越界全局内存访问、misaligned 访问、double free

# Race condition 检测
compute-sanitizer --tool racecheck ./my_cuda_app

# 初始化检测
compute-sanitizer --tool initcheck ./my_cuda_app
```

**实践建议**：
```cpp
// 1. 开发阶段始终启用 ASan
// CMakeLists.txt
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

// 2. CI 中跑 sanitizer 测试
// 3. CUDA 代码用 compute-sanitizer 跑 regression test
// 4. 生产中用 jemalloc/tcmalloc 的 heap profiler 检测内存增长
```

**【追问/扩展】**
- **ASan vs Valgrind 选择**：开发迭代用 ASan（快）；需要检查第三方库或不能重编译时用 Valgrind
- **`-fsanitize=address,undefined`**：可以同时启用多个 sanitizer
- **`ASAN_OPTIONS` 环境变量**：`ASAN_OPTIONS=detect_leaks=1:halt_on_error=0` 控制行为
- **在 PyTorch 开发中的应用**：PyTorch CI 跑 ASan 和 TSan 构建；CUDA kernel 的 bug 用 `compute-sanitizer` 检测
- **Core dump 分析**：`ulimit -c unlimited` 启用 core dump，用 `gdb ./my_app core` 分析崩溃现场

---
---

<!-- ================= SECTION_MARKER ================= -->

