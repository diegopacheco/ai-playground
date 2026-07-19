#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3.14 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python manage.py test

