#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
npm --prefix "$project_dir/sample" install
"$project_dir/sample/node_modules/.bin/playwright" install chromium
"$project_dir/start-runner.sh"
open -a "Google Chrome" "chrome://extensions"
open "$project_dir"
printf 'Enable Developer mode, select Load unpacked, and choose:\n%s\n' "$project_dir"
