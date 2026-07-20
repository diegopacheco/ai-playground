#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
open -a "Google Chrome" "chrome://extensions"
open "$project_dir"
printf 'Enable Developer mode, select Load unpacked, and choose:\n%s\n' "$project_dir"
