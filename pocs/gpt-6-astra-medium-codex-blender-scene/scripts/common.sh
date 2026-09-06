#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/.run/logs" "$ROOT/output"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if [ -z "${BLENDER_BIN:-}" ]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER_BIN="$(command -v blender)"
  elif [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then
    BLENDER_BIN=/Applications/Blender.app/Contents/MacOS/Blender
  else
    BLENDER_BIN=blender
  fi
fi
require() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required. Run ./scripts/setup.sh"; }
profile="${PROFILE:-standard}"
case "$profile" in
  draft) width=640; height=360; fps=12 ;;
  standard) width=1280; height=720; fps=24 ;;
  full) width=1920; height=1080; fps=24 ;;
  *) fail 'PROFILE must be draft, standard or full' ;;
esac
BUILD="$ROOT/output/$profile"
SCENE="$BUILD/frostbite-falls.blend"
mkdir -p "$BUILD"
