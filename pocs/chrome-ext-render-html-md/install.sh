#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/build.sh"

printf '\nChrome requires one confirmation for unpacked extensions.\n'
printf '1. Enable Developer mode.\n'
printf '2. Select Load unpacked.\n'
printf '3. Choose %s\n\n' "$ROOT/dist"
printf 'If it is already loaded, select its Reload button and refresh the GitHub tab.\n\n'

if [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Applications/Google Chrome.app" ]]; then
  open -a "Google Chrome" "chrome://extensions"
  open "$ROOT/dist"
elif command -v google-chrome >/dev/null 2>&1; then
  google-chrome "chrome://extensions" >/dev/null 2>&1 &
elif command -v google-chrome-stable >/dev/null 2>&1; then
  google-chrome-stable "chrome://extensions" >/dev/null 2>&1 &
fi
