#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require cargo
cd "$ROOT"
cargo clippy --release --all-targets --quiet -- -D warnings
cargo test --release
