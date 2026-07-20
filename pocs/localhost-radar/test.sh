#!/usr/bin/env bash
set -euo pipefail
node --test tests/*.test.js
python3 -m unittest discover -s tests -p 'test_*.py'
node -e "JSON.parse(require('fs').readFileSync('manifest.json', 'utf8')); JSON.parse(require('fs').readFileSync('native/com.diegopacheco.localhost_radar.json', 'utf8'))"
printf 'Localhost Radar checks passed.\n'
