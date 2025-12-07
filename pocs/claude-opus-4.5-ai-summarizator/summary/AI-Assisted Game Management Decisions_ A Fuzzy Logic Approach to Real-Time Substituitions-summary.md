# AI-Assisted Game Management Decisions: A Fuzzy Logic Approach to Real-Time Substituitions

**arXiv ID**: 2512.04480
**PDF**: https://arxiv.org/pdf/2512.04480.pdf

---

### Overview
The paper titled "AI-Assisted Game Management Decisions: A Fuzzy Logic Approach to Real-Time Substitutions" by Pedro Passos Farias introduces a Fuzzy Logic-based Decision Support System (DSS) tailored for optimizing player substitution decisions in elite soccer. With financial and competitive stakes at an all-time high, the system aims to replace intuition-based decisions with a more structured, objective framework. The DSS integrates a refined performance metric, termed PlayeRank, with physiological and contextual variables to derive a dynamic Substitution Priority (Pf final). Validation is conducted through a case study of the Brazil vs. Belgium match in the 2018 FIFA World Cup.

### Key Results
The study achieves significant findings:
- The model's output closely aligned with the substitution priorities recognized by expert coaching staff during the match, with several players flagged as high-priority for substitution according to the Fuzzy Logic system.
- Specific instances include:
  - Willian: Flagged with a priority score of 72.0 just before being substituted.
  - Gabriel Jesus: Rated with a critical priority of 99.1, aligning with his actual substitution shortly thereafter.
  - Fagner, marked with a maximum priority of 100.0 for substitution due to risking disciplinary issues and low performance, exemplified a key divergence from coach decisions, revealing the potential bias in human judgment.
- A notable point of conflict arose with Lukaku, whose performance metrics decline was detected 50 minutes earlier than when he was eventually substituted.

### Methodology
The methodology encompasses a robust analytical framework:
- **Dataset Composition**: The study utilized the "Soccer match event dataset," featuring 805,146 temporal observations from 3,035 unique players across 1,941 matches. Data types included match events and extensive player performance metrics.
- **Performance Metric Redefinition**: The PlayeRank metric was reformulated into a Cumulative Mean with Role-Aware Normalization to eliminate play-time exposure bias, enabling comparisons between players who spend differing amounts of time on the field.
- **Fuzzy Inference System (FIS)**: The Fuzzy Control System incorporates eight input variables that account for performance, fatigue, disciplinary risk, and tactical roles. Its architecture uses the Mamdani model to calculate a correction factor to a baseline performance metric. The outputs are designed to indicate urgency/priority for substitutions rather than mere predictions.

### Critical Insights
The paper addresses various critical insights and limitations:
- **Human Bias vs. Model Output**: The model's predictions highlighted flaws in human decision-making, especially regarding substitutions of players like Fagner, suggesting that cognitive biases such as sunk cost fallacies can impair real-time tactical decisions.
- **Performance Gaps**: The study identifies that traditional Machine Learning models that rely on historical data exhibit a “predictive ceiling,” achieving a maximum accuracy of 70%, contrasting sharply with the originality and flexibility of the Fuzzy Logic DSS that aims to surpass this limitation.
- **Utility of Explainability**: The fuzzy logic approach emphasizes interpretability, providing clear rationales behind substitution suggestions, in contrast to machine learning models, which often operate as "black boxes."
- **Limitations of Physical Assessment**: While performance metrics reflect player fatigue and effectiveness, the study noted that the current reliance on game data is an estimation rather than continuous biometric tracking, indicating a need for future enhancements to the model's accuracy.

In summary, the study demonstrates the potential of Fuzzy Logic in decision-making processes within sports analytics, particularly for player substitutions, combining systematic performance analysis with nuanced situational considerations.