#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "setup started"

require uv
[ -x "$PY" ] || uv venv -q --python 3.14 "$ROOT/.venv"
uv pip install -q --python "$PY" -r "$ROOT/requirements.txt"
"$PY" -m semif_poc.model || fail "model download failed"

if ! grep -q '^\.run/$' "$ROOT/.gitignore" 2>/dev/null; then
  printf ".run/\n" >>"$ROOT/.gitignore"
fi

log "setup done"
