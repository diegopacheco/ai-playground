#!/bin/bash
cargo build --release 2>&1 && echo "Starting Rustbird..." && ./target/release/superclaude-poc
