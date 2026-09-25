#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN not found" >&2
  exit 1
fi
if [ ! -x "$VENV_PY" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
"$VENV_PY" -m pip install -q --upgrade pip
CMAKE_ARGS="${CMAKE_ARGS:--DGGML_METAL=on}" "$VENV_PY" -m pip install -q -r requirements.txt
"$VENV_PY" -c "import sys; sys.path.insert(0, 'src'); from jev import load_model; load_model()"
echo "setup done: $("$VENV_PY" --version), model cached"
