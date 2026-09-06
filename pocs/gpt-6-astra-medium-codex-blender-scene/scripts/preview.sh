#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[ -f "$SCENE" ] || "$ROOT/scripts/build.sh"
mkdir -p "$ROOT/printscreens"
for second in 2 6 10 14; do
  "$BLENDER_BIN" --background "$SCENE" --python-exit-code 1 --render-output "$ROOT/printscreens/shot-" --render-format PNG --render-frame "$((second * fps))"
done
