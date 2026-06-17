#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-/tmp/task-api-bad-history}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

python3 "$HERE/generate_history.py" "$TARGET"

echo "repo ready at $TARGET"
echo "cd $TARGET && git log --oneline | head"
echo "then run /fix-git-history from that directory"
