#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash stop-all.sh
bash start-all.sh
