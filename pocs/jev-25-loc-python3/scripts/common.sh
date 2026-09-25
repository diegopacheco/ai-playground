#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.14}"
VENV_PY="$ROOT/.venv/bin/python"

require_venv() {
  if [ ! -x "$VENV_PY" ]; then
    echo "missing .venv, run ./scripts/setup.sh first" >&2
    exit 1
  fi
}
