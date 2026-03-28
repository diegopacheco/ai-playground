#!/bin/bash
cargo run -- --llm claude --model sonnet --refresh 5m --ui --port 3000 "$@"
