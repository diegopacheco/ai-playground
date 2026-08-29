#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release
echo "built target/release/monitor-stats"
