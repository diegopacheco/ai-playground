#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"

command -v cc >/dev/null 2>&1 || fail "cc not found, install the Xcode command line tools with: xcode-select --install"
command -v make >/dev/null 2>&1 || fail "make not found, install the Xcode command line tools with: xcode-select --install"

if ! command -v sdl2-config >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || fail "sdl2-config not found and Homebrew is missing, install SDL2 manually"
  echo "installing SDL2 with Homebrew"
  brew install sdl2
fi

echo "SDL2 $(sdl2-config --version)"
ensure_run_dirs
make all build/tests
echo "setup complete: $GAME_BIN"
