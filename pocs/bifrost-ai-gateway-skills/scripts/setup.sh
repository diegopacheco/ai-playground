#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require "$PYTHON"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 14) else 1)' || fail "$PYTHON must be Python 3.14"
log "python $("$PYTHON" -c 'import platform; print(platform.python_version())')"

require npx
bifrost_cmd -help 2>&1 | grep -- "-app-dir" >/dev/null || fail "could not download bifrost $BIFROST_VERSION"
log "bifrost $BIFROST_VERSION ready"

missing=0
for cli in $CLIS; do
  if command -v "$cli" >/dev/null 2>&1; then
    log "cli $cli found"
  else
    printf "WARN: cli %s not installed, its provider will return errors\n" "$cli" >&2
    missing=$((missing + 1))
  fi
done

if [ -f "$ROOT/.gitignore" ] && ! grep -q '^\.run/$' "$ROOT/.gitignore"; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi

if command -v ollama >/dev/null 2>&1 && ! ollama list >/dev/null 2>&1; then
  printf "WARN: the ollama server is not running, start it with: ollama serve\n" >&2
fi

log "setup done, $missing cli missing"
