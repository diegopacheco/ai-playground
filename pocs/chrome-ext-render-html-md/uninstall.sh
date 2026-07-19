#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$ROOT/dist"
rm -f "$ROOT/github-render.zip"

printf '\nSelect GitHub Render on the Chrome extensions page and choose Remove.\n'

if [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Applications/Google Chrome.app" ]]; then
  open -a "Google Chrome" "chrome://extensions"
elif command -v google-chrome >/dev/null 2>&1; then
  google-chrome "chrome://extensions" >/dev/null 2>&1 &
elif command -v google-chrome-stable >/dev/null 2>&1; then
  google-chrome-stable "chrome://extensions" >/dev/null 2>&1 &
fi
