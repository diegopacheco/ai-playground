# Arxiv Sumarizer

Arxiv Summarizer is a Rust application that fetches research papers from arXiv, extracts their content, and generates concise summaries using OpenAI's language models. The summaries are saved in markdown format for easy reading and sharing. The application try to fetch the latest 100 papers in the Artificial Intelligence (cs.AI) category.

<img src="logo-ras.png" width="400" />

## How it works?

* Build int Rust 1.90+
* Fetches papers about AI from arXiv `https://arxiv.org/list/cs.AI/recent`
* Make a prompt to call OpenAI API with the PDF content and the summary prompt.
* OpenAI model used: `gpt-4o-mini`
* Saves is in a markdown file.
* Uses the markdown files as cache to avoid reprocessing papers. 

## Build

```bash
cargo build
```

## Run 

```bash
OPEN_AI_API_KEY="sk-proj-..."
cargo run
```

Result:
```
❯ cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.09s
     Running `target/debug/arxiv-summarizer`
Found 99 existing summaries
Fetching papers from arXiv...
Found 100 papers
1 papers need processing
Processing: STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions
  Downloading PDF: STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions
  PDF saved: STELLA_ Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions.pdf
  Extracting text from PDF: STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions
  Generating summary: STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions
  Summary saved: STELLA_ Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions-summary.md
Progress: 1/1

Done!
```