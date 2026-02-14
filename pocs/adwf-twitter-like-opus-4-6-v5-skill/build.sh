#!/bin/bash
set -e
echo "Building backend..."
cd "$(dirname "$0")/backend"
cargo build 2>&1
cargo clippy 2>&1
echo "Backend build complete."
echo ""
echo "Building frontend..."
cd "$(dirname "$0")/frontend"
bun install 2>&1
bun run build 2>&1
echo "Frontend build complete."
echo ""
echo "Build finished successfully."
