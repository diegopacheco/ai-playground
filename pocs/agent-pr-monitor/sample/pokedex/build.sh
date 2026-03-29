#!/bin/bash
cd "$(dirname "$0")"
echo "=== Linting ==="
npx eslint src/ && echo "Lint OK"
echo ""
echo "=== Building ==="
npx ng build && echo "Build OK"
