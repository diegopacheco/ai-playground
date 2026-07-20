#!/usr/bin/env bash
set -euo pipefail
node --test tests/*.test.js
python3 -B -m unittest discover -s tests -p '*_test.py'
node -e "JSON.parse(require('fs').readFileSync('manifest.json', 'utf8'))"
printf 'FlowPrint checks passed.\n'
