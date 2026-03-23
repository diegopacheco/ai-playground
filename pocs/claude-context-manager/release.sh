#!/bin/bash
cargo build --release
cp target/release/claude-context-manager .
echo "Binary ready: ./claude-context-manager"
