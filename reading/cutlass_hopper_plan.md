# CUTLASS Hopper 阅读计划

写 Hopper kernel，从 [CUTLASS examples](https://github.com/NVIDIA/cutlass/tree/main/examples) 里抠思想。

**不要按 00→90 往下刷。** 大半是 Ampere/Turing 的 2.x API。Hopper 要看的是 CUTLASS 3.x：**TMA + WGMMA + warp specialization + cluster + persistent**。

官方声明这些 example **不是 benchmark**；测性能用 Profiler，或自己 autotune。

对照本仓库：

| 本仓库已有 | 对应 CUTLASS 里要看的 |
|:---|:---|
| `ptx_asm/06_hopper_wgmma_tma` | 48 / 49 的硬件原语 |
| `hopper_fa3/` | 88，两边对照 |
| `cutedsl_ref/` Layout / TiledCopy | 51 GETT |
| `profile/gemm/` TMA pipeline 调参 | 48 的 stages / smem |

CUTLASS 这边的价值：看工业实现怎么把这些原语拆成 **mainloop / epilogue / scheduler** 三件套，而不是再学一遍 `cp.async`。

---

## 怎么读（每个例程只抓 3 处）

1. `*_kernel` / `CollectiveMainloop`：TMA copy atom、pipeline、WGMMA atom
2. `CollectiveEpilogue`：C 怎么从寄存器出去、有没有 fusion
3. `tile_scheduler` / kernel schedule 枚举：谁负责下一个 tile

对着 ncu 想这几个问题，比对着模板想快：

- Producer 在等 TMA 还是 consumer 在等 MMA？把 `mbarrier` 的 arrive/wait 画成时间线。
- `Stages` 加 1，smem 涨多少、能否多藏一轮 TMA latency？
- Cluster size > 1 时 TMA multicast 省的是哪次 HBM 读？
- Persistent 比 one-tile-per-block 好在：launch 开销、L2 驻留、还是负载均衡？

`examples/cute/` 和 `test/unit/cute/core/` 补 Layout 代数；**kernel 思想还是 48 / 49 / 54 / 88**。

---

## 路径总览

```
第 1 段  48 → 49 → 54     Hopper GEMM：搬、算、精度
第 2 段  50 + 61          epilogue / fusion
第 3 段  88               对照自己的 FA3
第 4 段  57 + 55 + 67     grouped、mixed dtype、FP8 block scale
第 5 段  51               Layout 把非 GEMM 收成 GEMM
选读     56 / 63 / 53 / 52 / 68 / 113
```

下面清单打勾用。例程链接指向 NVIDIA/cutlass `main`。

---

## 第 1 段：Hopper GEMM 主干

这三条把骨架走完。后面都是变体。

- [ ] **[48 hopper_warp_specialized_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm)**  
  最小完整 Hopper GEMM。Producer warp 发 TMA，consumer warp 跑 WGMMA，`mbarrier` 同步，多 stage pipeline。先读懂这条。

- [ ] **[49 hopper_gemm_with_collective_builder](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder)**  
  Builder API + persistent 调度。mainloop / epilogue 怎么拼、几种 kernel schedule（warp-specialized persistent）差在哪。工业代码几乎都走这条，不是手写 48 那种全特化。

- [ ] **[54 hopper_fp8_warp_specialized_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm)**  
  FP8 Tensor Core。看 scale、accumulator 精度、和 48 的差异。Hopper 训练/推理的主力精度。

读这三条时盯：

1. 数据怎么进 SMEM：TMA descriptor、multicast、cluster
2. 谁算谁搬：warp group 划分、`setmaxnreg`（producer 少寄存器、consumer 多）
3. 怎么叠流水：pipeline stage 数 vs smem 容量 vs WGMMA 延迟
4. tile 怎么排：persistent 抢 tile，而不是 `grid = tiles`

---

## 第 2 段：Epilogue / Fusion

- [ ] **[50 hopper_gemm_with_epilogue_swizzle](https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle)**  
  自定义 mainloop + 向量化 epilogue + swizzle。算完之后怎么把寄存器里的 C 高效写回、避免 bank conflict / 非合并写。

- [ ] **[61 hopper_gemm_with_topk_and_softmax](https://github.com/NVIDIA/cutlass/tree/main/examples/61_hopper_gemm_with_topk_and_softmax)**  
  Epilogue fusion：GEMM 结果不落 HBM，直接 Top-K + softmax。融合的正确姿势是 visitor / epilogue，不是再 launch 一个 kernel。

- [ ] （选读）**[113 hopper_gemm_activation_fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/113_hopper_gemm_activation_fusion)**  
  GEMM + activation 融进同一 kernel。和 Ampere 的 12（bias+relu）同一思想，这是 3.x Hopper 写法。

---

## 第 3 段：FMHA，对照自己的 FA3

- [ ] **[88 hopper_fmha](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha)**  
  官方 Hopper FMHA：TMA 拉 QKV、WGMMA 做 QK/PV、online softmax 塞进 epilogue。

对照 `hopper_fa3/`：CUTLASS 版更框架化，Dao 的 FA3 更手搓 PTX。两边对照比只看一边快。

---

## 第 4 段：线上常见变体

不是「另一个 GEMM」，是调度 / 量化 / 不规则 batch。LLM serving、MoE、LoRA 会撞上。

先分清：

- **Batched / ptr-array**：`M,N,K` 相同，只是很多个
- **Grouped**：每个 problem 的 `M,N,K` 不同，scheduler 要处理负载不均

- [ ] **[57 hopper_grouped_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm)**  
  一组形状不同的 GEMM 打进同一个 kernel（MoE expert、变长序列）。

- [ ] **[55 hopper_mixed_dtype_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm)**  
  A/B 不同精度 + 融进 mainloop 的 dequant（INT8/FP8 weight、FP16 activation）。DeepGEMM / 量化 GEMM 的 CUTLASS 版。

- [ ] **[67 hopper_fp8_warp_specialized_gemm_with_blockwise_scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling)**  
  现代 FP8（不是 tensor-wise 一个 scale）怎么跟 MMA 对齐。

选读（第 4 段走完再看）：

- [ ] **[56 hopper_ptr_array_batched_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/56_hopper_ptr_array_batched_gemm)** — 同形状多 batch，指针数组，对比 57。
- [ ] **[68 hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling)** — 67 + grouped。
- [ ] **[69 hopper_mixed_dtype_grouped_gemm](https://github.com/NVIDIA/cutlass/tree/main/examples/69_hopper_mixed_dtype_grouped_gemm)** — 55 + grouped。
- [ ] **[63 hopper_gemm_with_weight_prefetch](https://github.com/NVIDIA/cutlass/tree/main/examples/63_hopper_gemm_with_weight_prefetch)** — 下一层 weight 提前用 TMA 拉。
- [ ] **[53 hopper_gemm_permute](https://github.com/NVIDIA/cutlass/tree/main/examples/53_hopper_gemm_permute)** — 输出按新 layout 写回，省一次 transpose/reshape。
- [ ] **[52 hopper_gather_scatter_fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/52_hopper_gather_scatter_fusion)** — gather 进 GEMM、scatter 出 GEMM。

---

## 第 5 段：CuTe Layout 接到真问题

- [ ] **[51 hopper_gett](https://github.com/NVIDIA/cutlass/tree/main/examples/51_hopper_gett)**  
  层次 Layout 把任意张量收缩变成 GEMM（GETT）。CuTe 的杀伤力在这，不是又一个 GEMM 变体。`cutedsl_ref` 里练的 Layout 接到真问题。

补 Layout 代数（卡住再翻，不必按周刷）：

- [CUTLASS `examples/cute/`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)
- [CUTLASS `test/unit/cute/core/`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core)

---

## 明确跳过

| 类别 | 例程 | 原因 |
|:---|:---|:---|
| 老架构 2.x | 00–46 绝大部分 | Volta/Turing/Ampere。`sgemm` / `cp.async` / WMMA 已经覆盖。 |
| Blackwell | 70–95、89–90、93 | `tcgen05`、TMEM，换卡才能用。原理以后扫 70/71。 |
| 接口 | 40 python、60 import | 不是 kernel 思想。 |
| 多卡 | 65 / 82 distributed_gemm | 等单卡 Hopper 流水熟了再看。 |
| 专用 | 62 sparse、111 SSD、quaternion、SYRK/TRMM、conv 系列 | 不是通用 HPC 主干。 |

老编号里只抽调度思想，有 Hopper 对应物就读 Hopper 的：

- Split-K / Stream-K：扫一眼 **[47](https://github.com/NVIDIA/cutlass/tree/main/examples/47_ampere_gemm_universal_streamk)** README 即可。Hopper 上更多是 persistent + cluster。
- 融合的问题定义：35 / 37 / 45 的 README 够用，实现去看 61 / 88 / 113。

---

## 和博客清单的配合

[blogs.md](blogs.md) 里这些可以并行当注释，不要替代读源码：

- CUTLASS 2.x & 3.x Intro（@BBuf）
- Hopper Mixed GEMM 的 CUTLASS 实现笔记（@BBuf）— 对着 55 读
- CUTLASS CuTe 实战（一）（二）（@进击的Killua）
- GEMM 流水线 single/multi-stage（@Titus）— 对着 48 的 stages 读
- cute Swizzle 系列 — 对着 50 读
