#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

APP_NAME="Meeting Limiter"
BUNDLE_ID="com.diegopacheco.meetinglimiter"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
VERSION="$(sed -n 's/.*"version": "\(.*\)".*/\1/p' "$ROOT/desktop/package.json" | head -1)"

"$SCRIPTS/uninstall.sh"
"$SCRIPTS/setup.sh"
"$SCRIPTS/make-icon.sh"

applications="/Applications"
[ -w "$applications" ] || { applications="$HOME/Applications"; mkdir -p "$applications"; }
bundle="$applications/$APP_NAME.app"
resources="$bundle/Contents/Resources"
plist="$bundle/Contents/Info.plist"

cp -R "$ROOT/desktop/node_modules/electron/dist/Electron.app" "$bundle"
rm -f "$resources/default_app.asar"
mkdir -p "$resources/app/desktop" "$resources/app/frontend/public"
cp "$ROOT/desktop/main.cjs" "$ROOT/desktop/preload.cjs" "$ROOT/desktop/boot.html" "$resources/app/desktop/"
cp "$ROOT/frontend/public/logo.svg" "$resources/app/frontend/public/"
printf '{"name":"meeting-limiter","version":"%s","main":"desktop/main.cjs"}\n' "$VERSION" > "$resources/app/package.json"
cp "$ROOT/desktop/icon.icns" "$resources/MeetingLimiter.icns"
printf '%s\n' "$ROOT" > "$resources/source-path"
printf '%s:%s\n' "$(dirname "$(command -v go)")" "$(dirname "$(command -v bun)")" > "$resources/tool-path"

/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $APP_NAME" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName $APP_NAME" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile MeetingLimiter.icns" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$plist"
touch "$bundle"
"$LSREGISTER" -f "$bundle"
log "$APP_NAME $VERSION installed at $bundle"
