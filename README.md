---
license: 
datasets:
- imagenet-1k
language:
- en
- zh
---

LiteVAR:
[arXiv](https://arxiv.org/abs/2411.17178)

Visual Autoregressive (VAR) has emerged as a promising approach in image generation, offering competitive potential and performance comparable to diffusion-based models. However, current AR-based visual generation models require substantial computational resources, limiting their applicability on resource-constrained devices. To address this issue, we conducted analysis and identified significant redundancy in three dimensions of the VAR model: (1) the attention map, (2) the attention outputs when using classifier free guidance, and (3) the data precision. Correspondingly, we proposed efficient attention mechanism and low-bit quantization method to enhance the efficiency of VAR models while maintaining performance. With negligible performance lost (less than 0.056 FID increase), we could achieve 85.2% reduction in attention computation, 50% reduction in overall memory and 1.5x latency reduction. To ensure deployment feasibility, we developed efficient training-free compression techniques and analyze the deployment feasibility and efficiency gain of each technique.

VAR:
[![arXiv](https://img.shields.io/badge/arXiv%20papr-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)[![demo platform](https://img.shields.io/badge/Play%20with%20VAR%21-VAR%20demo%20platform-lightblue)](https://var.vision/demo)

