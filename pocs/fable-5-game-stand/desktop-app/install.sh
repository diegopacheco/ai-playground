#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -f .desktop.pid ]; then
  pid="$(cat .desktop.pid)"
  kill "$pid" 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f .desktop.pid
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required"
  exit 1
fi
python3 -m venv .venv
.venv/bin/pip install -q -r ../requirements.txt
npm install
applications="/Applications"
if [ ! -w "$applications" ]; then
  applications="$HOME/Applications"
  mkdir -p "$applications"
fi
app_bundle="$applications/Game Stand.app"
rm -rf "$app_bundle"
cp -R node_modules/electron/dist/Electron.app "$app_bundle"
rm -f "$app_bundle/Contents/Resources/default_app.asar"
mkdir -p "$app_bundle/Contents/Resources/app"
cp main.js package.json "$app_bundle/Contents/Resources/app/"
cp assets/GameStand.icns "$app_bundle/Contents/Resources/GameStand.icns"
project_directory="$(cd .. && pwd)"
printf '%s\n' "$project_directory" > "$app_bundle/Contents/Resources/source-path"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName Game Stand" "$app_bundle/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName Game Stand" "$app_bundle/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.diegopacheco.gamestand" "$app_bundle/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile GameStand.icns" "$app_bundle/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString 1.1.0" "$app_bundle/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleVersion 1.1.0" "$app_bundle/Contents/Info.plist"
touch "$app_bundle"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$app_bundle"
echo "Game Stand installed at $app_bundle"
