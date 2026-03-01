#!/bin/bash
cd "$(dirname "$0")"

echo "=== Backend Tests ==="
cd backend
cargo test 2>&1
BACKEND_EXIT=$?
cd ..

echo ""
echo "=== Frontend Type Check ==="
cd frontend
bun install --silent 2>&1
npx tsc --noEmit 2>&1
FRONTEND_EXIT=$?
cd ..

echo ""
if [ $BACKEND_EXIT -eq 0 ] && [ $FRONTEND_EXIT -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
    exit 1
fi
