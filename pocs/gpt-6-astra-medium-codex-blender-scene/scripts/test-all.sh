#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
for script in "$ROOT"/scripts/*.sh; do bash -n "$script"; done
python3 -m unittest discover -s "$ROOT/tests" -v
[ -f "$SCENE" ] || "$ROOT/scripts/build.sh"
"$BLENDER_BIN" --background "$SCENE" --python-exit-code 1 --python "$ROOT/scene/verify.py"
if [ -f "$BUILD/frostbite-falls.mp4" ]; then
  python3 "$ROOT/scene/check_frames.py" "$BUILD"
  python3 "$ROOT/scene/verify_video.py" "$BUILD"
else
  fail 'Video verification requires ./scripts/run.sh first; no video checks were run'
fi
