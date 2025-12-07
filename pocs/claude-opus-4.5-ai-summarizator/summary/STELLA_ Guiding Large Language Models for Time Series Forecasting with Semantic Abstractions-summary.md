# STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions

**arXiv ID**: 2512.04871
**PDF**: https://arxiv.org/pdf/2512.04871.pdf

---

### Overview

The paper titled "STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions" introduces a novel framework named STELLA, which aims to leverage Large Language Models (LLMs) for time series forecasting by addressing the limitations of existing methodologies. The primary goal is to enhance the LLMs' capabilities by incorporating structured supplementary and complementary information derived from time series data. STELLA systematically mines and generates this information through a dynamic semantic abstraction mechanism, facilitating improved guidance for LLMs. The framework is evaluated across eight benchmark datasets for long- and short-term forecasting tasks, demonstrating its efficacy in zero-shot and few-shot settings.

### Key Results

STELLA consistently outperformed state-of-the-art forecasting methods across multiple datasets. In long-term forecasting, it achieved the best results in 60 evaluation settings. Key quantitative findings include:
- A relative reduction in Mean Squared Error (MSE) of 16.06% and 55.91% when compared to the state-of-the-art models PatchTST and Crossformer, respectively.
- A reduction of 20.25% in MSE compared to GPT4TS and 24.61% compared to MICN for long-term forecasting.
- In the M4 dataset evaluations for short-term forecasting, specific metrics used included symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE), and overall weighted average (OWA), although detailed scores in this domain were not explicitly provided.

### Methodology

The STELLA framework operates in three stages, starting with the structural representation of a time series, which is decomposed into trend, seasonal, and residual components using a Neural STL module. Each component is processed through a dual-path architecture:
1. A Temporal Convolutional Patch Encoder (TC-Patch) generates numerical embeddings.
2. The Semantic Anchor Module (SAM) creates two hierarchical prompts: a Corpus-level Semantic Prior (CSP) for global context and a Fine-grained Behavioral Prompt (FBP) for instance-specific guidance.

The framework undergoes rigorous evaluations on eight real-world benchmark datasets, using fixed input sequences and varying prediction horizons. Loss functions, including L1 and L2 loss, were applied for long-term forecasting, while SMAPE was used for short-term evaluations.

### Critical Insights

The study underscores the crucial role of supplementary and complementary information in enhancing LLM-based forecasting performance. One notable insight is that existing models often fail to capture the intrinsic patterns of time series data, leading to suboptimal forecasts. The authors highlight that static retrieval-based methods establish merely a correlational context instead of a generative understanding of time series dynamics. STELLA’s approach of dynamically generating semantic anchors addresses this gap, facilitating stronger model performance.

While the findings show impressive generalization capabilities and improved performance metrics across various tasks, the paper does not explicitly discuss any failure modes or hallucinations observed, nor does it detail performance inconsistencies across different domains. However, it does emphasize the need for ongoing research to bridge the modality gap and fully harness the power of LLMs in time series forecasting.