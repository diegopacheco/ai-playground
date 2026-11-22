#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
npx playwright screenshot --browser chromium --wait-for-selector '.tag.active' "$(service_url)" .run/view.png
echo ".run/view.png"
