# Semantic Drift Detector

A Rust CLI agent that runs the same prompt against an LLM repeatedly, embeds the outputs using TF-IDF term frequency vectors, computes cosine similarity against the baseline response, and plots semantic drift over time in the terminal. Detects when a model silently changes behavior.

## How it works

1. Sends a fixed prompt to the LLM via `claude` CLI
2. Tokenizes the response and builds a TF term-frequency embedding
3. Stores the response and embedding in SQLite (`drift.db`)
4. On subsequent runs, builds a shared vocabulary across all stored responses
5. Computes cosine similarity of each response against the first baseline
6. If similarity drops below `0.75`, drift is flagged
7. Plots an ASCII chart showing similarity over time

## Commands

```
cargo run -- run [prompt]      Probe the LLM and record response
cargo run -- report [prompt]   Show drift report without probing
cargo run -- history [prompt]  Show response history
```

## Output

```
=== Semantic Drift Plot (Cosine Similarity vs Baseline) ===

 1.00 |
 0.82 |                                                            X
 0.75 |
 0.71 | X
 0.50 |
      +------------------------------------------------------------
       Oldest                                                  Latest

  Legend: * = stable, X = drift detected (similarity < 0.75)

=== Drift Report ===

Date                     Similarity   Drifted?
----------------------------------------------
2026-03-18 06:45:47          0.6920        YES
2026-03-18 06:46:00          0.7314        YES
```

## Result

```
❯ ./run.sh
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.16s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
     Running `target/debug/agent-semantic-drift-detector run`
Probing LLM with prompt: "Explain what a hash map is and when you would use one. Be concise."
Got response (722 chars)
Saved record 925745aa-03e5-42e1-b32b-0fef4e8ae52c

=== Semantic Drift Plot (Cosine Similarity vs Baseline) ===

 1.00 |
 0.96 |
 0.93 |
 0.89 |
 0.86 |
 0.82 |
 0.79 |
 0.75 |                              X                             X
 0.71 | X
 0.68 |
 0.64 |
 0.61 |
 0.57 |
 0.54 |
 0.50 |
      +------------------------------------------------------------
       Oldest                                                  Latest

  Legend: * = stable, X = drift detected (similarity < 0.75)

=== Drift Report ===

Date                     Similarity   Drifted?
----------------------------------------------
2026-03-18 06:45:47          0.6920        YES
2026-03-18 06:46:00          0.7314        YES
2026-03-18 06:52:53          0.7462        YES
```

## Requirements

* Rust 1.93+ (edition 2024)
* `claude` CLI installed and configured

## Build and Run

```
./run.sh
```
