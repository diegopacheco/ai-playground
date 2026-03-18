#!/bin/bash
cargo build --release 2>&1 && ./target/release/agent-intent-eval "$@"
