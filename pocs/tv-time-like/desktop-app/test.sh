#!/usr/bin/env bash
set -e
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"
bash -n install.sh uninstall.sh run-desktop-app.sh test.sh
node --check build-icon.cjs
bun run test
