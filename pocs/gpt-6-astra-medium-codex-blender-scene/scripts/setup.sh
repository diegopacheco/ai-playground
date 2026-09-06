#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if ! command -v "$BLENDER_BIN" >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    command -v "$BLENDER_BIN" >/dev/null 2>&1 || brew install --cask blender
    command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg
  else
    fail 'Install Blender 4.5+ and FFmpeg with libx264, then run setup again. Set BLENDER_BIN if needed.'
  fi
fi
require "$BLENDER_BIN"
require ffmpeg
require ffprobe
require python3
"$BLENDER_BIN" --version | head -n 1
ffmpeg -version | head -n 1
printf 'Ready: %s, %sx%s, %s fps\n' "$profile" "$width" "$height" "$fps"
