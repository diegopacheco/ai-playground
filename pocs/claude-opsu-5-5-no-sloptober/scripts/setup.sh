#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"
require python3
require node
if ! command -v gh >/dev/null 2>&1; then
  log "gh not found, the no-slop-pr skill will only work with --text and --diff"
fi
chmod +x "$ROOT/.claude/skills/no-slop-pr/slop_check.py"
if ! grep -q '^\.run/$' "$ROOT/.gitignore"; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi
log "setup done"
