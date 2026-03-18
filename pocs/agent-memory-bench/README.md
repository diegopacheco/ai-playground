# Agent Memory Bench

A Rust-based benchmark that evaluates different memory strategies for AI agents. It measures how well an LLM can recall facts from interaction history using four distinct memory approaches across varying retrieval distances.

## Memory Strategies

| Strategy | Description |
|---|---|
| **Raw Context** | Feeds the full interaction history (or a windowed subset for large distances) directly into the prompt |
| **Summarization** | Groups interactions into batches of 50, summarizes per-person attributes, and queries against the summaries |
| **RAG** | Scores and retrieves the top-10 most relevant interactions based on person and attribute matching |
| **Knowledge Graph** | Builds a graph of person-attribute-value triples and queries the relevant subgraph plus neighbors |

## How It Works

1. Generates 1,000 random facts mapping people to favorite attributes (city, food, hobby, color)
2. For each retrieval distance (1, 10, 100, 1,000), picks a target fact and builds a prompt using each strategy
3. Calls Claude CLI to answer the recall question
4. Checks if the LLM response contains the expected answer
5. Reports per-strategy accuracy and latency

## Architecture

```
┌─────────────────────────────────────────────┐
│            Fact Generator (1000 facts)       │
│   Person ──► Attribute ──► Value             │
│   (Alice)    (food)        (sushi)           │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  ┌──────────┐ ┌────────┐ ┌─────────────────┐
  │Raw Context│ │  RAG   │ │ Knowledge Graph │
  │Summarize │ │ Top-K  │ │    Triples      │
  └─────┬────┘ └───┬────┘ └───────┬─────────┘
        │          │              │
        ▼          ▼              ▼
  ┌─────────────────────────────────────────┐
  │          Claude CLI (sonnet)            │
  │     "What is Alice's favorite food?"    │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────┐
  │   Accuracy + Latency Report             │
  └─────────────────────────────────────────┘
```

## Retrieval Distances

The benchmark tests recall at four distances to simulate how well each strategy handles temporal decay:

- **Distance 1** — The target fact is the most recent interaction
- **Distance 10** — The target fact is 10 interactions ago
- **Distance 100** — The target fact is 100 interactions ago
- **Distance 1000** — The target fact is the very first interaction (hardest)

## Requirements

- Rust (edition 2021)
- Claude CLI (`claude`) installed and configured
- Dependencies: `tokio`, `serde`, `serde_json`, `rand`

## Build and Run

```bash
./run.sh
```

## Result

```
❯ ./run.sh
    Finished `release` profile [optimized] target(s) in 0.02s
=== Agent Memory Benchmark ===

Strategy             Distance   Correct    Latency(ms)  Response                                 Expected
----------------------------------------------------------------------------------------------------------------
raw_context          1          YES        10775        paella                                   paella
summarization        1          NO         18301        Risotto                                  paella
rag                  1          NO         10675        risotto                                  paella
knowledge_graph      1          NO         120006       ERROR: LLM timed out after 120s          paella
raw_context          10         YES        5706         pho                                      pho
summarization        10         NO         6570         curry                                    pho
rag                  10         NO         7341         curry                                    pho
knowledge_graph      10         YES        5968         pho                                      pho
raw_context          100        NO         6041         ceviche                                  paella
summarization        100        NO         11692        ceviche                                  paella
rag                  100        YES        6832         paella                                   paella
knowledge_graph      100        NO         5459         risotto                                  paella
raw_context          1000       YES        5522         Sydney                                   Sydney
summarization        1000       NO         17699        Bangkok                                  Sydney
rag                  1000       NO         5791         Prague                                   Sydney
knowledge_graph      1000       NO         5838         Lisbon                                   Sydney

=== Summary ===

Strategy             Correct    Total      Accuracy
--------------------------------------------------
raw_context          3          4          75.0%
summarization        0          4          0.0%
rag                  1          4          25.0%
knowledge_graph      1          4          25.0%

=== Benchmark Complete ===
```

### Key Takeaways

| Strategy | Accuracy | Observation |
|---|---|---|
| **Raw Context** | 75.0% | Best overall — direct context wins when it fits the window |
| **Summarization** | 0.0% | Lossy compression discards specific values during aggregation |
| **RAG** | 25.0% | Keyword-based retrieval helps at medium distances but struggles with ambiguity |
| **Knowledge Graph** | 25.0% | Structured lookup works when the graph has the right node, fails on sparse subgraphs |
