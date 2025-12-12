#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
file="$root_dir/index.html"
if [[ ! -f "$file" ]]; then
  echo "index.html not found in $root_dir"
  exit 1
fi
if command -v open >/dev/null 2>&1; then
  open "$file"
  exit 0
fi
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$file"
  exit 0
fi
echo "No supported browser opener found"
exit 1
