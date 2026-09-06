#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require "$BLENDER_BIN"
[ -f "$SCENE" ] || "$ROOT/scripts/build.sh"
exec "$BLENDER_BIN" "$SCENE"
