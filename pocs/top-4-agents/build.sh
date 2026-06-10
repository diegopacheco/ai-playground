#!/usr/bin/env bash
set -euo pipefail
cargo build --release
echo "built target/release/agentop"
