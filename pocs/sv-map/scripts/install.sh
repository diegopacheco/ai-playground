#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

APP_NAME="SV Map"
BUNDLE_ID="com.diegopacheco.svmap"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
VERSION="$(node -p "require('$ROOT/package.json').version")"

"$SCRIPTS/uninstall.sh"
"$SCRIPTS/setup.sh"

applications="/Applications"
[ -w "$applications" ] || { applications="$HOME/Applications"; mkdir -p "$applications"; }
bundle="$applications/$APP_NAME.app"
resources="$bundle/Contents/Resources"
plist="$bundle/Contents/Info.plist"

cp -R "$ROOT/node_modules/electron/dist/Electron.app" "$bundle"
rm -f "$resources/default_app.asar"
mkdir -p "$resources/app/desktop" "$resources/app/web"
cp "$ROOT/desktop/main.cjs" "$ROOT/desktop/preload.cjs" "$ROOT/desktop/boot.html" "$resources/app/desktop/"
cp "$ROOT/web/logo.svg" "$resources/app/web/"
printf '{"name":"sv-map","version":"%s","main":"desktop/main.cjs"}\n' "$VERSION" > "$resources/app/package.json"
cp "$ROOT/desktop/icon.icns" "$resources/SVMap.icns"
printf '%s\n' "$ROOT" > "$resources/source-path"
dirname "$(command -v node)" > "$resources/node-dir"

/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $APP_NAME" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName $APP_NAME" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile SVMap.icns" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$plist"
touch "$bundle"
"$LSREGISTER" -f "$bundle"
log "$APP_NAME $VERSION installed at $bundle"
