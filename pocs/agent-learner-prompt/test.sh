#!/bin/bash
cd "$(dirname "$0")"
echo "Building project..."
cargo build --release
if [ $? -ne 0 ]; then
    echo "Build FAILED"
    exit 1
fi
echo "Build OK"
echo ""
echo "Running tests..."
cargo test
if [ $? -ne 0 ]; then
    echo "Tests FAILED"
    exit 1
fi
echo "Tests OK"
echo ""
echo "Testing CLI help..."
./target/release/agent-learner --help
echo ""
echo "Testing --show-memory..."
./target/release/agent-learner --show-memory
echo ""
echo "Testing --show-anti-patterns..."
./target/release/agent-learner --show-anti-patterns
echo ""
echo "Testing --list-prompts..."
./target/release/agent-learner --list-prompts
echo ""
echo "All tests passed"
