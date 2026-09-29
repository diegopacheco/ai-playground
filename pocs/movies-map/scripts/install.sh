#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

APP_NAME="Movies Map"
BUNDLE_ID="com.diegopacheco.moviesmap"
INSTALL_DIR="${MOVIES_MAP_INSTALL_DIR:-/Applications}"
TARGET="$INSTALL_DIR/$APP_NAME.app"
ELECTRON_APP="$ROOT/node_modules/electron/dist/Electron.app"

require node
[ -d "$ELECTRON_APP" ] || fail "electron is missing, run scripts/setup.sh"
[ -f "$ROOT/data/movies.json" ] || fail "data/movies.json is missing, run scripts/setup.sh"

"$SCRIPTS/uninstall.sh"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
BUNDLE="$STAGING/$APP_NAME.app"
PLIST="$BUNDLE/Contents/Info.plist"
cp -R "$ELECTRON_APP" "$BUNDLE"
mv "$BUNDLE/Contents/MacOS/Electron" "$BUNDLE/Contents/MacOS/$APP_NAME"
plutil -replace CFBundleDisplayName -string "$APP_NAME" "$PLIST"
plutil -replace CFBundleName -string "$APP_NAME" "$PLIST"
plutil -replace CFBundleExecutable -string "$APP_NAME" "$PLIST"
plutil -replace CFBundleIdentifier -string "$BUNDLE_ID" "$PLIST"
plutil -replace CFBundleShortVersionString -string 1.0.0 "$PLIST"
plutil -replace CFBundleVersion -string 1 "$PLIST"
plutil -replace LSApplicationCategoryType -string public.app-category.entertainment "$PLIST"
plutil -remove ElectronAsarIntegrity "$PLIST" 2>/dev/null || true

ICONSET="$STAGING/icon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$ROOT/logo.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$ROOT/logo.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$BUNDLE/Contents/Resources/moviesmap.icns"
plutil -replace CFBundleIconFile -string moviesmap.icns "$PLIST"
rm -f "$BUNDLE/Contents/Resources/electron.icns" "$BUNDLE/Contents/Resources/default_app.asar"

APP_DIR="$BUNDLE/Contents/Resources/app"
mkdir -p "$APP_DIR/desktop" "$APP_DIR/web"
printf '{ "name": "movies-map", "productName": "%s", "version": "1.0.0", "main": "desktop/main.cjs" }\n' "$APP_NAME" > "$APP_DIR/package.json"
cp "$ROOT/desktop/main.cjs" "$ROOT/desktop/preload.cjs" "$ROOT/desktop/boot.html" "$APP_DIR/desktop/"
cp "$ROOT/web/logo.svg" "$APP_DIR/web/"
printf '{ "root": "%s", "nodeDir": "%s" }\n' "$ROOT" "$(dirname "$(command -v node)")" > "$APP_DIR/desktop/config.json"

codesign --force --deep --sign - "$BUNDLE" 2>/dev/null
cp -R "$BUNDLE" "$TARGET"
touch "$TARGET"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$TARGET"
log "installed $TARGET"
