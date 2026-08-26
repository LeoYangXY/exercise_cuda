# NVIDIA PTX ISA 中文翻译（结合 CUDA 背景改写）

本目录是 [NVIDIA PTX ISA 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 的中文翻译，由 AI 对照原文、结合 CUDA 背景改写而成，供学习参考。

## 如何在线观看（一键看）

本项目已托管在 GitHub，开启 GitHub Pages 后即可通过以下链接直接在线浏览（无需下载）：

```
https://leoyangxy.github.io/learn-cuda-cute-triton/nv_docs/isa-cn/
```

> 如果上方链接打不开，说明仓库 owner 还未在 GitHub 网页开启 Pages（见下方「开启 GitHub Pages」）。

### 开启 GitHub Pages（仓库 owner 操作，只需一次）

1. 打开 https://github.com/LeoYangXY/learn-cuda-cute-triton → **Settings** → **Pages**
2. **Source** 选择 `Deploy from a branch`
3. **Branch** 选择 `master`，目录选 `/ (root)`
4. 点击 **Save**

等待 1~2 分钟，访问上面的链接即可。之后每次 push 新内容，Pages 会自动更新。

## 本地查看（备选）

直接用浏览器打开 `index.html` 即可（或部署到任意静态服务器）。

如果本地查看图片/样式有问题，可在本目录起一个静态服务器：

```bash
python3 -m http.server 8000
# 然后访问 http://localhost:8000/
```

## 目录结构

- `index.html` —— 总目录，点击进入各章
- `chapter1.html` ~ `chapter14.html` —— 14 章正文 + Notices
- `images/` —— 文档中的原图（本地化，离线可用）

## 翻译约定

- **专有技术名词保留英文原词不翻译**：如 `tensor`、`warp`、`cluster`、`atomic`、`mma`、`wgmma`、`transformer`、`state space`、`.shared`/`.global` 等修饰符。
- **指令语法、寄存器、代码示例保留英文原文**。
- **自然语言讲解文字译为中文**，并结合 CUDA 背景解释其含义。

> 本翻译仅供参考，权威内容请以 NVIDIA 官方英文文档为准。
