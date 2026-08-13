#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v python3.14 >/dev/null || { printf 'Python 3.14.6 is required\n' >&2; exit 1; }
runtime="$(python3.14 -c 'import platform; print(platform.python_version())')"
[[ "$runtime" == "3.14.6" ]] || { printf 'Python 3.14.6 is required, found %s\n' "$runtime" >&2; exit 1; }
python3.14 main.py "${1:-Should we use graph engineering for an AI workflow?}" "${@:2}"
