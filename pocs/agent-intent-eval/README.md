# Agent Intent Eval

Rust-based intent preservation evaluation tool that measures how consistently an LLM responds to semantically equivalent prompts. It generates 10 paraphrases of a base prompt, queries the LLM for each, then uses an LLM-as-judge approach to score pairwise semantic similarity between all responses.

## How it works

1. Takes a base prompt (default: "What is the capital of France?")
2. Generates 10 paraphrased versions of the prompt
3. Queries the Claude CLI for each paraphrase
4. Uses an LLM judge to compare each response against the baseline (first response)
5. Computes an overall robustness score (0.0 to 1.0)
6. Produces a verdict: HIGH (>=0.8), MODERATE (>=0.5), or LOW (<0.5) robustness

## Requirements

* Rust (edition 2024)
* Claude CLI installed and available in PATH

## How to build and run
```bash
./run.sh
```

## Custom prompt
```bash
./run.sh "What is the meaning of life?"
```

## Result
```
❯ ./run.sh
    Finished `release` profile [optimized] target(s) in 0.02s
Intent Preservation Eval
Base prompt: What is the capital of France?
Generating 10 paraphrases and querying LLM...

[1/10] Querying: What is the capital of France?
[1/10] Got response (6 chars)
[2/10] Querying: Can you tell me: What is the capital of France?
[2/10] Got response (6 chars)
[3/10] Querying: I'd like to know: What is the capital of France?
[3/10] Got response (6 chars)
[4/10] Querying: Please explain: What is the capital of France?
[4/10] Got response (6 chars)
[5/10] Querying: In your own words, what is the capital of france?
[5/10] Got response (6 chars)
[6/10] Querying: Help me understand: What is the capital of France?
[6/10] Got response (6 chars)
[7/10] Querying: Could you clarify: What is the capital of France?
[7/10] Got response (6 chars)
[8/10] Querying: What is your take on: what is the capital of france?
[8/10] Got response (6 chars)
[9/10] Querying: Describe for me: what is the capital of france?
[9/10] Got response (6 chars)
[10/10] Querying: Give me a detailed answer: what is the capital of france?
[10/10] Got response (6 chars)

Judging pairwise semantic similarity...
  Comparing pair (1, 2)...
  Comparing pair (1, 3)...
  Comparing pair (1, 4)...
  Comparing pair (1, 5)...
  Comparing pair (1, 6)...
  Comparing pair (1, 7)...
  Comparing pair (1, 8)...
  Comparing pair (1, 9)...
  Comparing pair (1, 10)...

================================================================================
INTENT PRESERVATION EVALUATION REPORT
================================================================================

--- Paraphrase 1 ---
Prompt: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 2 ---
Prompt: Can you tell me: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 3 ---
Prompt: I'd like to know: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 4 ---
Prompt: Please explain: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 5 ---
Prompt: In your own words, what is the capital of france?
Response (first 200 chars): Paris.

--- Paraphrase 6 ---
Prompt: Help me understand: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 7 ---
Prompt: Could you clarify: What is the capital of France?
Response (first 200 chars): Paris.

--- Paraphrase 8 ---
Prompt: What is your take on: what is the capital of france?
Response (first 200 chars): Paris.

--- Paraphrase 9 ---
Prompt: Describe for me: what is the capital of france?
Response (first 200 chars): Paris.

--- Paraphrase 10 ---
Prompt: Give me a detailed answer: what is the capital of france?
Response (first 200 chars): Paris.

--------------------------------------------------------------------------------
PAIRWISE SIMILARITY SCORES
--------------------------------------------------------------------------------
  Pair (1, 2): 1.00
  Pair (1, 3): 1.00
  Pair (1, 4): 1.00
  Pair (1, 5): 1.00
  Pair (1, 6): 1.00
  Pair (1, 7): 1.00
  Pair (1, 8): 1.00
  Pair (1, 9): 1.00
  Pair (1, 10): 1.00

--------------------------------------------------------------------------------
OVERALL ROBUSTNESS SCORE: 1.00
--------------------------------------------------------------------------------
VERDICT: HIGH robustness - LLM gives semantically consistent answers across phrasings
================================================================================
```
