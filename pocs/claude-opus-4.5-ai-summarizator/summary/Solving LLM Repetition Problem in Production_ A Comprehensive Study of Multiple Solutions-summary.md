# Solving LLM Repetition Problem in Production: A Comprehensive Study of Multiple Solutions

**arXiv ID**: 2512.04419
**PDF**: https://arxiv.org/pdf/2512.04419.pdf

---

### Overview
The paper titled "Solving LLM Repetition Problem in Production: A Comprehensive Study of Multiple Solutions" addresses the critical repetition problem faced by Large Language Models (LLMs) in production environments. This issue is notably evident in batch code interpretation tasks, where models generate repetitive outputs without proper termination, leading to significant performance degradation and system stalling. The authors identify three specific repetition patterns (business rule generation, method call relationship analysis, and PlantUML diagram syntax generation) and propose various strategies, grounded in theoretical analysis and validated through extensive experiments, to mitigate this problem.

### Key Results
The paper reports several key quantitative findings:
1. **Repetition Problem Impact**: The occurrence of repetition increased processing time in batch scenarios by 43% to 471%, with a reproducibility rate of 75-80% across different deployment modes.
2. **Beam Search Solution Effectiveness**: Implementing Beam Search decoding with `early_stopping=True` eliminated the repetition problem, achieving a 0% repetition rate, while `early_stopping=False` resulted in a 60% repetition rate.
3. **Presence Penalty**: Applying a `presence_penalty` of 1.2 effectively resolved repetition issues for BadCase 1 (business rule generation).
4. **Direct Preference Optimization (DPO)**: Implementing DPO fine-tuning uniformly addressed all three BadCases, yielding significant reductions in repetition rates and restoring normal processing times.

### Methodology
The authors conducted a systematic analysis of the repetition problem using actual deployment scenarios for LLMs in batch processing of financial transactions. The methodology involved:
- Identifying three distinct types of repetition.
- Applying a theoretical framework grounded in Markov models to analyze the underlying causes of repetition.
- Heavily utilizing experimental validation to assess the effectiveness of the proposed solutions, including Beam Search decoding, the presence penalty, and DPO fine-tuning.
- The experimental environment included comparisons across different model deployment settings, employing two modes for the vLLM (LoRA-enabled and Merged model) and examining performance metrics across various runs.

### Critical Insights
Several nuances and limitations were highlighted:
- **Greedy Decoding Limitations**: The study emphasizes that greedy decoding is inherently prone to repetition due to a single-path exploration strategy and no long-term planning, which amplifies self-reinforcement.
- **Parameter Sensitivity**: The effectiveness of Beam Search is heavily dependent on strict adherence to parameter configuration, especially the necessity of setting `early_stopping=True`.
- **Task-Specific Solutions**: While the presence penalty is effective for BadCase 1, it does not generalize to BadCase 2 and BadCase 3, thus indicating a need for tailored solutions rather than a one-size-fits-all approach.
- **Cost of DPO Fine-tuning**: DPO, while providing a universal solution, is resource-intensive and requires careful dataset construction, making it less accessible for quick deployment compared to post-hoc solutions like Beam Search. 

Overall, the paper provides practical, production-ready solutions for addressing critical repetition issues in LLM applications, contributing valuable insights for the deployment of LLMs in contexts demanding high-quality and deterministic outputs.