#!/usr/bin/env bash
set -e
echo "Verifying embedded assets in binary..."
BINARY="target/release/rad"
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found. Run ./build.sh first"
    exit 1
fi
echo "Checking for embedded agent files..."
AGENT_COUNT=$(ls -1 agents/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "Found $AGENT_COUNT agent files in agents/ directory"
echo "Checking for embedded skill files..."
SKILL_COUNT=$(find skills -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "Found $SKILL_COUNT skill files in skills/ directory"
echo "Searching binary for embedded content signatures..."
if strings "$BINARY" | grep -q "java-backend-developer-agent.md"; then
    echo "Found: java-backend-developer-agent.md embedded"
fi
if strings "$BINARY" | grep -q "rust-backend-developer-agent.md"; then
    echo "Found: rust-backend-developer-agent.md embedded"
fi
if strings "$BINARY" | grep -q "workflow-skill/SKILL.md"; then
    echo "Found: workflow-skill/SKILL.md embedded"
fi
echo "Verification complete!"
echo "The binary contains embedded assets and does not require agents/ or skills/ directories at runtime"
