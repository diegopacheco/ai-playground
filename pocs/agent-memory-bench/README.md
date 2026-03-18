# Agent Memory Bench

A Rust-based benchmark that evaluates different memory strategies for AI agents. It measures how well an LLM can recall facts from interaction history using four distinct memory approaches across varying retrieval distances.

## Memory Strategies

* **Raw Context** - Feeds the full interaction history (or a windowed subset for large distances) directly into the prompt
* **Summarization** - Groups interactions into batches, summarizes per-person attributes, and queries against the summaries
* **RAG** - Scores and retrieves the top-K most relevant interactions based on person and attribute matching
* **Knowledge Graph** - Builds a graph of person-attribute-value triples and queries the relevant subgraph plus neighbors

## How It Works

1. Generates 1000 random facts mapping people to favorite attributes (city, food, hobby, color)
2. For each retrieval distance (1, 10, 100, 1000), picks a target fact and builds a prompt using each strategy
3. Calls Claude via CLI to answer the recall question
4. Checks if the LLM response contains the expected answer
5. Reports per-strategy accuracy and latency

## Requirements

* Rust (edition 2021)
* Claude CLI (`claude`) installed and configured
* tokio, serde, serde_json, rand

## Build and Run

```bash
./run.sh
```

## Output

Prints a table with strategy, distance, correctness, latency, response, and expected answer, followed by an accuracy summary per strategy.

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