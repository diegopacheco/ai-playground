#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v oak >/dev/null
test -d .oak
test "$(tr -d '\r\n' < message.txt)" = "version=2"
oak info
oak status --short
oak log --oneline
