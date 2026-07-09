#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TOPIC="${*:-Durable execution with Temporal}"
node src/client.ts "$TOPIC"
