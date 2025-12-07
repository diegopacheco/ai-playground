# LLMs Know More Than Words: A Genre Study with Syntax, Metaphor & Phonetics

**arXiv ID**: 2512.04957
**PDF**: https://arxiv.org/pdf/2512.04957.pdf

---

## Overview

The paper titled "LLMs Know More Than Words: A Genre Study with Syntax, Metaphor & Phonetics" explores the extent to which large language models (LLMs) can capture deeper linguistic features from raw text. The authors introduce a multilingual genre classification dataset derived from Project Gutenberg, encompassing works in six languages (English, French, German, Italian, Spanish, and Portuguese). The dataset is designed for evaluating models' abilities to understand and leverage syntactic structures, metaphor usage, and phonetic metrics to improve genre classification between different forms of literary writing—specifically, poetry, novels, and dramas.

## Key Results

The experiments yielded significant results across various models and genre pairs:

1. **Model Performance**: 
   - The baseline model BERT achieved F1 scores of 0.97 in English for the Poetry vs. Novel task, while for the more challenging Novel vs. Drama task, its F1 score was lower at 0.90.
   - The models tended to perform best in English and Portuguese, with average F1 scores for Poetry vs. Novel exceeding 0.80 across nearly all languages. In contrast, French exhibited lower variability in performance, particularly with BERT and DistilBERT.
  
2. **Feature Contributions**:
   - Incorporating metrical patterns (metre pattern) consistently improved performance, often yielding increases of 2-7% in F1 scores, particularly in the Poetry vs. Novel and Poetry vs. Drama tasks.
   - On the other hand, syntactic tree depth and metaphor counts only produced minor improvements, highlighting that phonetic characteristics provided stronger performance enhancements.

3. **Across Languages**:
   - The highest performance was consistently observed in English and Portuguese, while French displayed the greatest variability in performance and significant challenges differentiating genres. 

## Methodology

The study's methodology involves constructing a multilingual dataset containing approximately 45,000 sentences. The dataset is drawn from publicly available texts in Project Gutenberg, with sentences tailored for three binary genre classification tasks: Poetry vs. Novel, Drama vs. Poetry, and Drama vs. Novel. 

- **Evaluation Process**: The LLMs were evaluated based on F1 scores computed for each classification task. Three explicit linguistic features were incorporated into the models to analyze their effects on classification accuracy. The training involved using baseline models such as BERT, DistilBERT, RoBERTa, and Metaphor-RoBERTa, with and without the linguistic features.

## Critical Insights

Some critical insights and observations from the study include:

1. **Feature Effects**: 
   - The metrically based features demonstrated the capability to enhance classification tasks significantly, reaffirming their role in capturing the distinctiveness of poetic language.
   - Syntax tree information and metaphor counts did not yield substantial gains overall, suggesting that while these features can be useful, they may not be as effective in directly influencing genre classification.

2. **Model Limitations**:
   - The reliance on the Project Gutenberg corpus may bias the findings due to its focus on canonical literature, potentially underrepresenting modern and marginalized voices.
   - While the study indicates that LLMs can leverage linguistic signals for enhanced comprehension, the authors acknowledge possible limitations in fully capturing complex linguistic nuances with the current feature extraction methods.

3. **Variability Across Languages**: 
   - The performance results highlighted that not all languages exhibited the same levels of proficiency with genre distinctions. For instance, models struggled with tasks in French due to overlapping linguistic features between genres, revealing a performance gap based on linguistic diversity and characteristics inherent to each language.

This investigation sets a foundation for future exploration into how diverse linguistic dimensions can shape LLM performance, particularly in genre classification tasks across languages and contexts.