#!/bin/bash
set -e
cd "$(dirname "$0")"
BIN=target/release/port-doctor
if [ ! -x "$BIN" ]; then
  cargo build --release
fi
exec "$BIN" "$@"
