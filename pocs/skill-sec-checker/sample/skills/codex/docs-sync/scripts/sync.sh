#!/usr/bin/env bash
set -euo pipefail
mkdir -p site/docs
rsync -a --delete --itemize-changes docs/ site/docs/ | grep -c '^>f' || true
