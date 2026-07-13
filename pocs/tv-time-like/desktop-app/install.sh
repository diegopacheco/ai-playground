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
echo "Installed $target"
