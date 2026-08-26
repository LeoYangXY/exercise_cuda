# NVIDIA PTX ISA 中文翻译（结合 CUDA 背景改写）

本目录是 [NVIDIA PTX ISA 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 的中文翻译，由 AI 对照原文、结合 CUDA 背景改写而成，供学习参考。

## 如何使用

直接用浏览器打开 `index.html` 即可（或部署到 GitHub Pages / 任意静态服务器）。

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
