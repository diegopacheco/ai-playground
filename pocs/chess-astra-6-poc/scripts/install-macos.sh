#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MACOS="$ROOT/macos"
APP_NAME="Wizards Gambit"
BUNDLE_ID="com.diegopacheco.wizardsgambit"
TARGET="/Applications/$APP_NAME.app"

[ "$(uname -s)" = "Darwin" ] || { printf 'ERROR: macOS is required\n' >&2; exit 1; }
for tool in node npm npx bun; do
  command -v "$tool" >/dev/null 2>&1 || { printf 'ERROR: %s is required\n' "$tool" >&2; exit 1; }
done

"$ROOT/scripts/uninstall-macos.sh"
[ -d "$ROOT/node_modules/vite" ] || "$ROOT/scripts/setup.sh"
(cd "$MACOS" && npm install --no-audit --no-fund)
node "$MACOS/build-icon.mjs"
cp "$ROOT/public/logo.svg" "$MACOS/app/logo.svg"
node -e 'require("fs").writeFileSync(process.argv[1], JSON.stringify({ root: process.argv[2], path: process.argv[3] }, null, 2))' "$MACOS/app/config.json" "$ROOT" "$PATH"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

arch="$(uname -m)"
[ "$arch" = "x86_64" ] && arch="x64"
(cd "$MACOS" && npx @electron/packager . "$APP_NAME" \
  --platform=darwin \
  --arch="$arch" \
  --icon="$MACOS/build/icon.icns" \
  --app-bundle-id="$BUNDLE_ID" \
  --app-version="$(node -p 'require("./package.json").version')" \
  --out="$STAGE" \
  --ignore='^/(build|tests|node_modules|package-lock\.json|build-icon\.mjs|playwright\.config\.ts)($|/)' \
  --asar \
  --overwrite)

ditto "$STAGE/$APP_NAME-darwin-$arch/$APP_NAME.app" "$TARGET"
codesign --force --deep --sign - "$TARGET" >/dev/null 2>&1
printf 'Installed %s\nOpen it from Launchpad, Spotlight, or: open "%s"\n' "$TARGET" "$TARGET"
