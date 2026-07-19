#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST="$ROOT/dist"

rm -rf "$DIST"
rm -f "$ROOT/github-render.zip"
mkdir -p "$DIST/icons"
node "$ROOT/scripts/generate-icons.js" "$DIST/icons"
cp "$ROOT/manifest.json" "$DIST/manifest.json"
cp "$ROOT/src/background.js" "$DIST/background.js"
cp "$ROOT/src/content.js" "$DIST/content.js"
cp "$ROOT/src/content.css" "$DIST/content.css"
cp "$ROOT/src/html-viewer.html" "$DIST/html-viewer.html"
cp "$ROOT/src/html-viewer.js" "$DIST/html-viewer.js"
cp "$ROOT/src/renderer.js" "$DIST/renderer.js"
cp "$ROOT/src/popup.html" "$DIST/popup.html"
cp "$ROOT/src/popup.css" "$DIST/popup.css"
cp "$ROOT/src/popup.js" "$DIST/popup.js"
cd "$DIST"
zip -qr "$ROOT/github-render.zip" .
printf 'Built %s\n' "$DIST"
printf 'Packed %s\n' "$ROOT/github-render.zip"
