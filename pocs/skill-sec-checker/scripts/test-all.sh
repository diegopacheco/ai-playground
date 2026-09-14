#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

for f in install.sh uninstall.sh sample/build.sh scripts/*.sh; do
  bash -n "$f" || { echo "syntax error in $f" >&2; exit 1; }
done
python3 -m unittest discover -s "$ROOT/tests" -v
