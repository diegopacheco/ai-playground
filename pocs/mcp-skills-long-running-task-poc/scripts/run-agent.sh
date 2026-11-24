#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun
require claude
REPORT_AGENT_MODEL="${2:-claude-sonnet-5}" bun client/main.ts "${1:-electric bikes europe}"
