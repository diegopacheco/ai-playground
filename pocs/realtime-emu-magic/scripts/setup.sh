#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
node -e 'if (Number(process.versions.node.split(".")[0]) < 24) process.exit(1)'
npm ci --ignore-scripts
npx playwright install chromium
node --env-file-if-exists=.env.local --input-type=module -e 'import { agentStatus } from "./agent.mjs"; const status = await agentStatus(); console.log(status.message); if (!status.available) process.exit(1)'
printf 'Setup complete.\n'
