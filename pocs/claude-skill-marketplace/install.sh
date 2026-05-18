#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SCOPE="${CSM_SCOPE:-user}"

if ! command -v claude >/dev/null 2>&1; then
  echo "error: 'claude' CLI not found on PATH" >&2
  exit 1
fi

if claude plugin marketplace list 2>/dev/null | grep -q "^claude-skill-marketplace\b"; then
  echo "marketplace already registered; refreshing from source..."
  claude plugin marketplace update claude-skill-marketplace
else
  claude plugin marketplace add "$SRC_DIR" --scope "$SCOPE"
fi

echo ""
echo "marketplace: claude-skill-marketplace"
echo "scope:       $SCOPE"
echo "source:      $SRC_DIR"
echo ""
echo "next: inside Claude Code run:"
echo "  /plugin install hello-world@claude-skill-marketplace"
echo ""
echo "or from terminal:"
echo "  claude plugin install hello-world@claude-skill-marketplace"
