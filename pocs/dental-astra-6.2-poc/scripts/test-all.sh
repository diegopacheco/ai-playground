#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export ATLAS_URL="$(service_url)"
npx playwright test
