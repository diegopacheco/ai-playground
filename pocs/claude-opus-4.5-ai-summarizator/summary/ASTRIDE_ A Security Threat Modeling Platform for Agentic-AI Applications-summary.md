# ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications

**arXiv ID**: 2512.04785
**PDF**: https://arxiv.org/pdf/2512.04785.pdf

---

### Overview
The paper introduces ASTRIDE, an automated threat modeling platform developed specifically for AI agent-based applications. ASTRIDE extends the traditional STRIDE threat modeling framework by adding a new category for AI agent-specific attacks, which encompasses unique vulnerabilities such as prompt injection, unsafe tool invocation, and reasoning subversion. The primary goal of ASTRIDE is to enhance the efficacy of threat modeling in modern AI systems by incorporating advanced vision-language models (VLMs) and reasoning via the OpenAI-gpt-oss model, which automates the analysis directly from visual architecture diagrams such as data flow diagrams.

### Key Results
The evaluations demonstrate that ASTRIDE significantly improves threat identification accuracy. Specific quantitative metrics include:
- The training of three fine-tuned VLMs—Llama-Vision, Pix2Struct, and Qwen2-VL—on approximately 1,200 annotated threat modeling diagrams resulted in enhanced prediction capabilities.
- The evaluation of VLMs showed an increase in detected vulnerabilities post-fine-tuning, where for instance, Llama-Vision expanded from identifying only prompt injection to recognizing context poisoning and unsafe tool invocation, resulting in actionable mitigations.
- The reasoning capabilities of the OpenAI-gpt-oss model were validated via comparative analysis against individual predictions from VLMs, showing effective synthesis and refinement of threat models.

### Methodology
ASTRIDE's approach is underpinned by a consortium of fine-tuned VLMs that analyze complex visual representations of system architectures to detect threats. The dataset used consists of labeled threat modeling diagrams that include:
- Data Flow Diagrams associated with threat vectors and trust boundaries.
- Approximately 1,200 records were synthetically generated and annotated with trust boundaries, component interactions, and mitigation strategies, structured in a format conducive to training the models.
The evaluation process involved splitting the dataset into training (2/3), validation (1/6), and testing (1/6) subsets, allowing for a comprehensive assessment of model performance during training cycles.

### Critical Insights
The integration of ASTRIDE revealed several nuanced observations:
- The fine-tuned VLMs exhibited a marked improvement in their ability to detect AI-specific threats post-training, reflecting the effectiveness of fine-tuning on specialized datasets.
- There were indications of overfitting in some models, as evidenced by the validation loss exceeding training loss, suggesting room for further optimization.
- The paper notes that ASTRIDE successfully reduces dependency on human expertise through automation but admits a potential limitation in handling conflicting outputs from various models, which requires careful orchestration by the reasoning layer to synthesize a coherent threat model.
- The necessity of continuous updates to model training data is implied, as emerging threats in AI systems evolve rapidly.

In conclusion, ASTRIDE represents an innovative step forward in automated threat modeling for AI applications, merging fine-tuned VLMs with advanced reasoning to address a critical gap in the security landscape of modern software architectures.