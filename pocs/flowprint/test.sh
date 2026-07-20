#!/usr/bin/env bash
set -euo pipefail
node --test tests/*.test.js
node -e "JSON.parse(require('fs').readFileSync('manifest.json', 'utf8'))"
printf 'FlowPrint checks passed.\n'
