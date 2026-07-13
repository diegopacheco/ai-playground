#!/usr/bin/env bash
set -e
root="$(cd "$(dirname "$0")" && pwd)"
project="$(cd "$root/.." && pwd)"
install_dir="${REELMARK_INSTALL_DIR:-/Applications}"
target="$install_dir/Reelmark.app"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cd "$root"
bun install
bun run build
if [ ! -d "$root/node_modules/electron/dist/Electron.app" ]; then
  node "$root/node_modules/electron/install.js"
fi
bundle="$staging/Reelmark.app"
cp -R "$root/node_modules/electron/dist/Electron.app" "$bundle"
mv "$bundle/Contents/MacOS/Electron" "$bundle/Contents/MacOS/Reelmark"
plutil -replace CFBundleDisplayName -string Reelmark "$bundle/Contents/Info.plist"
plutil -replace CFBundleExecutable -string Reelmark "$bundle/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string com.diegopacheco.reelmark "$bundle/Contents/Info.plist"
plutil -replace CFBundleName -string Reelmark "$bundle/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string 1.0.0 "$bundle/Contents/Info.plist"
plutil -replace CFBundleVersion -string 1 "$bundle/Contents/Info.plist"
plutil -replace LSApplicationCategoryType -string public.app-category.entertainment "$bundle/Contents/Info.plist"
plutil -remove ElectronAsarIntegrity "$bundle/Contents/Info.plist" 2>/dev/null || true
iconset="$staging/reelmark.iconset"
mkdir -p "$iconset"
sips -z 16 16 "$root/reelmark-icon.png" --out "$iconset/icon_16x16.png" >/dev/null
sips -z 32 32 "$root/reelmark-icon.png" --out "$iconset/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$root/reelmark-icon.png" --out "$iconset/icon_32x32.png" >/dev/null
sips -z 64 64 "$root/reelmark-icon.png" --out "$iconset/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$root/reelmark-icon.png" --out "$iconset/icon_128x128.png" >/dev/null
sips -z 256 256 "$root/reelmark-icon.png" --out "$iconset/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$root/reelmark-icon.png" --out "$iconset/icon_256x256.png" >/dev/null
sips -z 512 512 "$root/reelmark-icon.png" --out "$iconset/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$root/reelmark-icon.png" --out "$iconset/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$root/reelmark-icon.png" --out "$iconset/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$iconset" -o "$bundle/Contents/Resources/reelmark.icns"
plutil -replace CFBundleIconFile -string reelmark.icns "$bundle/Contents/Info.plist"
rm -f "$bundle/Contents/Resources/electron.icns"
rm -rf "$bundle/Contents/Resources/default_app.asar" "$bundle/Contents/Resources/app.asar"
mkdir -p "$bundle/Contents/Resources/app"
cp "$root/main.cjs" "$root/package.json" "$bundle/Contents/Resources/app/"
cp -R "$project/dist" "$bundle/Contents/Resources/app/web"
cp -R "$project/server" "$bundle/Contents/Resources/app/server"
cp -R "$project/shared" "$bundle/Contents/Resources/app/shared"
cp -R "$project/prompts" "$bundle/Contents/Resources/app/prompts"
plutil -create xml1 "$bundle/Contents/Resources/app/config.plist"
plutil -insert DatabasePath -string "$project/data/reelmark.db" "$bundle/Contents/Resources/app/config.plist"
codesign --force --deep --sign - "$bundle"
if [ -w "$install_dir" ]; then
  rm -rf "$target"
  cp -R "$bundle" "$target"
else
  sudo rm -rf "$target"
  sudo cp -R "$bundle" "$target"
fi
touch "$target"
if [ "$install_dir" = "/Applications" ]; then
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$target"
  killall Dock 2>/dev/null || true
fi
echo "Installed $target"
