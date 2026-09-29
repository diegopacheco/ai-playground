#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
qlmanage -t -s 1024 -o "$WORK" "$ROOT/web/logo.svg" >/dev/null
cp "$WORK/logo.svg.png" "$ROOT/desktop/icon.png"
mkdir -p "$WORK/icon.iconset"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$ROOT/desktop/icon.png" --out "$WORK/icon.iconset/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$ROOT/desktop/icon.png" --out "$WORK/icon.iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$WORK/icon.iconset" -o "$ROOT/desktop/icon.icns"
echo "icon written to desktop/icon.icns"
