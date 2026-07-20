#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
"$project_dir/stop-runner.sh"
open -a "Google Chrome" "chrome://extensions"
printf 'Select FlowPrint and choose Remove.\n'
