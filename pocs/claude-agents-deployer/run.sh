#!/bin/bash
cd "$(dirname "$0")"
mkdir -p target/test
cp -r agents target/test/
cp target/release/claude-agents-deployer target/test/
cd target/test
./claude-agents-deployer
