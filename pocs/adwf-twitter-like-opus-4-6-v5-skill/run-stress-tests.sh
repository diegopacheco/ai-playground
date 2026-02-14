#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
k6 run "$SCRIPT_DIR/tests/stress/load-test.js" 2>&1
