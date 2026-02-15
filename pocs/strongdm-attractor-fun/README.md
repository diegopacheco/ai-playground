# Attractor

⏺ Attractor is a DOT-graph-based pipeline runner for orchestrating multi-stage AI workflows.

  What it does

  You define a pipeline as a DOT graph (the same format used by Graphviz), where each node is a processing step and edges define
  the execution order. The system then walks the graph topologically, executing each node in sequence.

  digraph pipeline {
      ask [type="llm" prompt="Summarize this code"];
      check [type="condition" condition="$ask_output contains error"];
      fix [type="llm" prompt="Fix the error: ${ask_output}"];
      done [type="transform" operation="uppercase"];
      ask -> check -> fix -> done;
  }

  Each node can be one of 8 types:
  - llm - calls an LLM (OpenAI, Anthropic, Gemini)
  - tool - runs a shell command or agent tool
  - transform - manipulates data (regex, json extract, template, uppercase, etc.)
  - condition - branches based on state ($var == "value", and/or)
  - human - prompts a human for input via stdin
  - parallel_start/end - runs nodes concurrently
  - subgraph - scoped sub-pipelines

  State flows between nodes automatically via ${node_output} variable references.

  The 3-layer architecture

  1. llm-client - unified API across OpenAI/Anthropic/Gemini (same request/response types regardless of provider)
  2. agent-loop - agentic tool-use loop (LLM calls tools, sees results, loops until done)
  3. attractor - the pipeline runner itself, built on top of both

  What's the point

  The core idea: define AI workflows as graphs, not code. Instead of writing bespoke orchestration logic for each multi-step AI
  task, you declare the pipeline shape in a .dot file and let the runner handle execution order, state passing, parallelism, and
  branching.

  Use cases: chained LLM calls (generate -> review -> refine), data processing pipelines with AI steps, human-in-the-loop approval
  workflows, conditional branching based on LLM outputs.

  Is it unique?

  Not really. It sits in the same space as:
  - LangChain/LangGraph - Python, graph-based LLM orchestration
  - Prefect/Airflow - DAG-based workflow runners (but not AI-specific)
  - DSPy - programmatic LLM pipelines
  - Semantic Kernel - Microsoft's AI orchestration

  What makes it somewhat distinctive:
  1. DOT as the pipeline language - a well-known, simple, widely-tooled graph format (you can visualize your pipeline with
  Graphviz)
  2. Rust, minimal dependencies - only 5 external crates, no framework bloat
  3. Stylesheet separation - model config (temperature, model choice, system prompts) is separated from pipeline logic via a JSON
  stylesheet, similar to CSS separating style from HTML
  4. Built-in agent loop - not just chained API calls, but a full tool-using agent that can read/write files, run commands, etc.

  But the fundamental concept - DAG-based AI workflow orchestration - is a crowded space. The differentiator here is simplicity and
   the DOT format choice rather than a fundamentally new paradigm.

## Result

```
❯   export ATTRACTOR_MODEL="gpt-4o-mini"
❯ cargo run --bin attractor-cli -- pipeline.dot
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
     Running `target/debug/attractor-cli pipeline.dot`
Pipeline completed successfully.
Final state:
  ask_output = Rust is a systems programming language known for its performance and safety. Here are three of its top benefits:

1. **Memory Safety Without Garbage Collection**: Rust ensures memory safety through its ownership model, which enforces strict rules about how memory is accessed and managed at compile time. This eliminates common programming errors such as null pointer dereferences and buffer overflows, without requiring a garbage collector. This leads to safer code and better performance as no runtime overhead for garbage collection is needed.

2. **Concurrency and Performance**: Rust’s design makes it easier to write concurrent code safely. The language's type system and ownership model help prevent data races, which are common issues in concurrent programming. As a result, developers can take advantage of multi-core processors without sacrificing safety. Rust also offers high performance comparable to that of C and C++, making it suitable for systems-level programming and applications requiring high efficiency.

3. **Strong Type System and Modern Features**: Rust boasts a powerful type system and modern programming features such as pattern matching, algebraic data types, and traits. This allows for expressive code that can be easier to maintain and reason about. The language also includes a robust package manager (Cargo) that facilitates dependency management and project organization, making it simpler to build and share libraries.

Overall, Rust's emphasis on safety, performance, and developer experience makes it a compelling choice for a variety of applications, from systems programming to web development and beyond.
  last_output = SUMMARY: RUST IS A SYSTEMS PROGRAMMING LANGUAGE KNOWN FOR ITS PERFORMANCE AND SAFETY. HERE ARE THREE OF ITS TOP BENEFITS:

1. **MEMORY SAFETY WITHOUT GARBAGE COLLECTION**: RUST ENSURES MEMORY SAFETY THROUGH ITS OWNERSHIP MODEL, WHICH ENFORCES STRICT RULES ABOUT HOW MEMORY IS ACCESSED AND MANAGED AT COMPILE TIME. THIS ELIMINATES COMMON PROGRAMMING ERRORS SUCH AS NULL POINTER DEREFERENCES AND BUFFER OVERFLOWS, WITHOUT REQUIRING A GARBAGE COLLECTOR. THIS LEADS TO SAFER CODE AND BETTER PERFORMANCE AS NO RUNTIME OVERHEAD FOR GARBAGE COLLECTION IS NEEDED.

2. **CONCURRENCY AND PERFORMANCE**: RUST’S DESIGN MAKES IT EASIER TO WRITE CONCURRENT CODE SAFELY. THE LANGUAGE'S TYPE SYSTEM AND OWNERSHIP MODEL HELP PREVENT DATA RACES, WHICH ARE COMMON ISSUES IN CONCURRENT PROGRAMMING. AS A RESULT, DEVELOPERS CAN TAKE ADVANTAGE OF MULTI-CORE PROCESSORS WITHOUT SACRIFICING SAFETY. RUST ALSO OFFERS HIGH PERFORMANCE COMPARABLE TO THAT OF C AND C++, MAKING IT SUITABLE FOR SYSTEMS-LEVEL PROGRAMMING AND APPLICATIONS REQUIRING HIGH EFFICIENCY.

3. **STRONG TYPE SYSTEM AND MODERN FEATURES**: RUST BOASTS A POWERFUL TYPE SYSTEM AND MODERN PROGRAMMING FEATURES SUCH AS PATTERN MATCHING, ALGEBRAIC DATA TYPES, AND TRAITS. THIS ALLOWS FOR EXPRESSIVE CODE THAT CAN BE EASIER TO MAINTAIN AND REASON ABOUT. THE LANGUAGE ALSO INCLUDES A ROBUST PACKAGE MANAGER (CARGO) THAT FACILITATES DEPENDENCY MANAGEMENT AND PROJECT ORGANIZATION, MAKING IT SIMPLER TO BUILD AND SHARE LIBRARIES.

OVERALL, RUST'S EMPHASIS ON SAFETY, PERFORMANCE, AND DEVELOPER EXPERIENCE MAKES IT A COMPELLING CHOICE FOR A VARIETY OF APPLICATIONS, FROM SYSTEMS PROGRAMMING TO WEB DEVELOPMENT AND BEYOND.
  output_output = SUMMARY: RUST IS A SYSTEMS PROGRAMMING LANGUAGE KNOWN FOR ITS PERFORMANCE AND SAFETY. HERE ARE THREE OF ITS TOP BENEFITS:

1. **MEMORY SAFETY WITHOUT GARBAGE COLLECTION**: RUST ENSURES MEMORY SAFETY THROUGH ITS OWNERSHIP MODEL, WHICH ENFORCES STRICT RULES ABOUT HOW MEMORY IS ACCESSED AND MANAGED AT COMPILE TIME. THIS ELIMINATES COMMON PROGRAMMING ERRORS SUCH AS NULL POINTER DEREFERENCES AND BUFFER OVERFLOWS, WITHOUT REQUIRING A GARBAGE COLLECTOR. THIS LEADS TO SAFER CODE AND BETTER PERFORMANCE AS NO RUNTIME OVERHEAD FOR GARBAGE COLLECTION IS NEEDED.

2. **CONCURRENCY AND PERFORMANCE**: RUST’S DESIGN MAKES IT EASIER TO WRITE CONCURRENT CODE SAFELY. THE LANGUAGE'S TYPE SYSTEM AND OWNERSHIP MODEL HELP PREVENT DATA RACES, WHICH ARE COMMON ISSUES IN CONCURRENT PROGRAMMING. AS A RESULT, DEVELOPERS CAN TAKE ADVANTAGE OF MULTI-CORE PROCESSORS WITHOUT SACRIFICING SAFETY. RUST ALSO OFFERS HIGH PERFORMANCE COMPARABLE TO THAT OF C AND C++, MAKING IT SUITABLE FOR SYSTEMS-LEVEL PROGRAMMING AND APPLICATIONS REQUIRING HIGH EFFICIENCY.

3. **STRONG TYPE SYSTEM AND MODERN FEATURES**: RUST BOASTS A POWERFUL TYPE SYSTEM AND MODERN PROGRAMMING FEATURES SUCH AS PATTERN MATCHING, ALGEBRAIC DATA TYPES, AND TRAITS. THIS ALLOWS FOR EXPRESSIVE CODE THAT CAN BE EASIER TO MAINTAIN AND REASON ABOUT. THE LANGUAGE ALSO INCLUDES A ROBUST PACKAGE MANAGER (CARGO) THAT FACILITATES DEPENDENCY MANAGEMENT AND PROJECT ORGANIZATION, MAKING IT SIMPLER TO BUILD AND SHARE LIBRARIES.

OVERALL, RUST'S EMPHASIS ON SAFETY, PERFORMANCE, AND DEVELOPER EXPERIENCE MAKES IT A COMPELLING CHOICE FOR A VARIETY OF APPLICATIONS, FROM SYSTEMS PROGRAMMING TO WEB DEVELOPMENT AND BEYOND.
  summarize_output = Summary: Rust is a systems programming language known for its performance and safety. Here are three of its top benefits:

1. **Memory Safety Without Garbage Collection**: Rust ensures memory safety through its ownership model, which enforces strict rules about how memory is accessed and managed at compile time. This eliminates common programming errors such as null pointer dereferences and buffer overflows, without requiring a garbage collector. This leads to safer code and better performance as no runtime overhead for garbage collection is needed.

2. **Concurrency and Performance**: Rust’s design makes it easier to write concurrent code safely. The language's type system and ownership model help prevent data races, which are common issues in concurrent programming. As a result, developers can take advantage of multi-core processors without sacrificing safety. Rust also offers high performance comparable to that of C and C++, making it suitable for systems-level programming and applications requiring high efficiency.

3. **Strong Type System and Modern Features**: Rust boasts a powerful type system and modern programming features such as pattern matching, algebraic data types, and traits. This allows for expressive code that can be easier to maintain and reason about. The language also includes a robust package manager (Cargo) that facilitates dependency management and project organization, making it simpler to build and share libraries.

Overall, Rust's emphasis on safety, performance, and developer experience makes it a compelling choice for a variety of applications, from systems programming to web development and beyond.
❯ ls
Cargo.lock         crates             pipeline-tool.dot  target
Cargo.toml         pipeline-local.dot pipeline.dot
❯ cat pipeline.dot
digraph pipeline {
    ask [type="llm" prompt="What are the top 3 benefits of Rust?"];
    summarize [type="transform" operation="template" prompt="Summary: ${ask_output}"];
    output [type="transform" operation="uppercase" prompt="${summarize_output}"];

    ask -> summarize;
    summarize -> output;
}
```

```
  ---
  pipeline-interesting.dot - Code Analyzer (6 nodes, no API key needed)

  scan ──> count_lines ──> list_tests ──> count_tests ──> report_template ──> shout

  Runs 4 shell commands to scan the codebase (count .rs files, count lines, find all tests, count tests per file), then chains the
  results into a template, and uppercases the final report. Output shows: 42 Rust files, 22 tests across 8 files.

  ---
  pipeline-condition.dot - Health Check (7 nodes, no API key needed)

  check_rust ──> check_cargo ──> check_git ──> disk_usage ──> build_test ──> json_payload ──> final_report

  Checks your toolchain versions, disk usage of target/, runs cargo test, builds a JSON payload from all results, then formats a
  final report. Detected: rustc 1.93.0, 645M target dir, all 22 tests passing.

  ---
  To run them yourself:

  cargo run --bin attractor-cli -- pipeline-interesting.dot
  cargo run --bin attractor-cli -- pipeline-condition.dot
```