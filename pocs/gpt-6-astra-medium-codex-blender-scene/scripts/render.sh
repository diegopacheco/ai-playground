#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require "$BLENDER_BIN"
[ -f "$SCENE" ] || "$ROOT/scripts/build.sh"
mkdir -p "$BUILD/frames"
exec "$BLENDER_BIN" --background "$SCENE" --python-exit-code 1 --render-output "$BUILD/frames/" --render-format PNG --render-anim
