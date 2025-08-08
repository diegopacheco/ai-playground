#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
	python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "\n[install-deps] Virtual environment ready: $VENV_DIR"
echo "[install-deps] To use it manually: source $VENV_DIR/bin/activate"
