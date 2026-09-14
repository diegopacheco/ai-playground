#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || { echo "python3 3.9 or newer is required" >&2; exit 1; }
chmod +x "$ROOT"/install.sh "$ROOT"/uninstall.sh "$ROOT"/scripts/*.sh "$ROOT"/sample/build.sh
"$ROOT/sample/build.sh"
echo "setup complete: python3 $(python3 -c 'import platform; print(platform.python_version())'), no dependencies to install"
