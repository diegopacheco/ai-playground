#!/bin/bash
set -e
cd "$(dirname "$0")"
cargo build --release
echo "built: target/release/port-doctor"
