# Agent Semantic Similarity Eval

A Rust-based evaluation service that scores how semantically similar a candidate answer is to a golden (reference) answer. It combines three independent scoring strategies into a weighted final score.

## Scoring Strategies

| Strategy | Weight | Description |
|----------|--------|-------------|
| **Cosine Similarity** | 25% | TF-IDF based cosine similarity between tokenized texts |
| **BERTScore (LLM-approximated)** | 35% | Uses an LLM to approximate BERTScore precision, recall, and F1 |
| **LLM Judge** | 40% | An LLM acts as an expert judge evaluating factual accuracy, completeness, and semantic equivalence |

## Verdicts

| Final Score | Verdict |
|-------------|---------|
| >= 0.8 | PASS - Strong semantic match |
| >= 0.6 | MARGINAL - Partial match, review recommended |
| < 0.6 | FAIL - Weak semantic match |

## Tech Stack

- **Rust** with Axum web framework
- **Claude CLI** as the LLM backend (model: sonnet)
- **Tokio** async runtime

## API Endpoints

### POST /api/eval
Evaluate a single answer.

Request:
```json
{
  "question": "What is Rust?",
  "golden_answer": "Rust is a systems programming language focused on safety and performance",
  "candidate_answer": "Rust is a language that emphasizes memory safety without garbage collection"
}
```

Response:
```json
{
  "question": "What is Rust?",
  "golden_answer": "Rust is a systems programming language focused on safety and performance",
  "candidate_answer": "Rust is a language that emphasizes memory safety without garbage collection",
  "cosine_score": 0.45,
  "bert_score": 0.82,
  "judge_score": 0.85,
  "judge_reasoning": "Both answers correctly describe Rust's focus on safety",
  "final_score": 0.74,
  "verdict": "MARGINAL - Partial match, review recommended"
}
```

### POST /api/eval/batch
Evaluate multiple answers at once.

Request:
```json
{
  "items": [
    {
      "question": "What is Rust?",
      "golden_answer": "A systems programming language",
      "candidate_answer": "A language for safe systems code"
    }
  ]
}
```

Response includes all individual results plus `average_score`, `pass_count`, `fail_count`, and `marginal_count`.

## How to Run

### Prerequisites
- Rust toolchain
- Claude CLI installed and configured

### Run
```bash
./run.sh
```
Server starts on `http://0.0.0.0:3000`.
