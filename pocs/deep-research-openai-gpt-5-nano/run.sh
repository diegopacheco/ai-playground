#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
	source .venv/bin/activate
fi

python src/main.py
