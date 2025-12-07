# Generative AI for Self-Adaptive Systems: State of the Art and Research Roadmap

**arXiv ID**: 2512.04680
**PDF**: https://arxiv.org/pdf/2512.04680.pdf

---

### Overview

The paper titled **"Generative AI for Self-Adaptive Systems: State of the Art and Research Roadmap"** aims to explore the integration of generative artificial intelligence (GenAI) within self-adaptive systems (SASs). It specifically highlights the potential benefits and challenges of using GenAI technologies, such as large language models (LLMs) and diffusion models, to enhance the core functionalities of SASs characterized by the MAPE-K feedback loop (monitoring, analyzing, planning, execution, and shared knowledge). The primary goal is to provide a comprehensive understanding that can guide researchers and practitioners in leveraging GenAI to improve the autonomy of SASs and enhance human-system interactions.

### Key Results

The paper outlines potential applications of GenAI within SASs, categorizing benefits into two main areas:

1. **Enhancements of MAPE-K Functions**: It identifies enhancements in:
   - **Monitoring**: Effective log parsing and anomaly detection with reported parsing accuracy reaching up to **95%** (compared to previous state-of-the-art methods).
   - **Analysis and Planning**: Use of LLMs to reason about system adaptations and improve decision-making processes. For example, LLM-based approaches to time series analysis claimed superior performance compared to traditional models.
   - **Knowledge Sharing**: LLMs can facilitate knowledge management through structured knowledge graphs, translating natural language into domain-specific applications.

2. **Human-on-the-Loop Settings**: The study highlights the importance of integrating humans into the adaptation process to enhance autonomy through user preference acquisition, system transparency, and collaboration insights.

The research also delineates several categories of literature reviewed, including **219 pieces** spanning various functionalities and focusing on MAPE-K and human-centered approaches.

### Methodology

The authors conducted a systematic literature search across key conferences in the fields of self-adaptive systems, software engineering, artificial intelligence, and human-computer interaction. The methodology involved:

- **Literature Selection**: A total of **5,874 papers** were originally collected based on keywords such as Transformers, LLMs, and generative approaches. After applying stringent relevance criteria—filtering for direct relevance to GenAI and SAS—the final selection included **219 papers**.
- **Categorization**: The selected papers were organized into relevant categories based on MAPE-K functionalities and human-on-the-loop (HOTL) settings, ensuring a comprehensive overview of how GenAI can contribute to various elements within SASs.

### Critical Insights

The paper notes several challenges for integrating GenAI into SASs:

- **Limited Research in SAS**: There is a scarcity of direct literature linking GenAI with self-adaptive systems, with a need to explore adjacent fields due to limited publications in leading conferences.
- **Technological Diversity and Rapid Evolution**: The multifaceted nature of SAS methodologies and the fast-paced developments in GenAI technologies complicate the synthesis of a cohesive understanding, indicating that ongoing research must continuously adapt.
- **Inherent Limitations of GenAI**: Specific behaviors observed include instances of determinism in generative outputs (e.g., LLMs’ outputs can lack consistency), which may impact reliability in critical applications.
- **Performance Gaps Across Domains**: The effectiveness of GenAI systems may vary significantly across different operational contexts, necessitating domain-specific strategies for implementation and evaluation.

These insights underline the importance of continued exploration and the necessity of developing robust integration strategies for GenAI technologies in self-adaptive systems. The paper concludes with a research roadmap that identifies potential avenues for future exploration and addresses current shortcomings in GenAI applications within SAS frameworks.