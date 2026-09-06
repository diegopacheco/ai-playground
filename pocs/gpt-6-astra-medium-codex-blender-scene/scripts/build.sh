#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require "$BLENDER_BIN"
"$BLENDER_BIN" --background --factory-startup --python-exit-code 1 --python "$ROOT/scene/build.py" -- --output "$SCENE" --width "$width" --height "$height" --fps "$fps"
python3 "$ROOT/scene/soundtrack.py" "$BUILD/soundtrack.wav"
