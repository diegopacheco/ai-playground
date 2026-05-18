#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  echo "error: 'claude' CLI not found on PATH" >&2
  exit 1
fi

if claude plugin list 2>/dev/null | grep -q "hello-world"; then
  claude plugin uninstall hello-world 2>&1 || true
fi

if claude plugin marketplace list 2>/dev/null | grep -q "^claude-skill-marketplace\b"; then
  claude plugin marketplace remove claude-skill-marketplace
  echo "marketplace removed"
else
  echo "marketplace not registered"
fi
