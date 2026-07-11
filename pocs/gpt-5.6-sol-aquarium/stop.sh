#!/usr/bin/env bash
if [ -f /tmp/pacificarium.pid ]; then
  kill "$(cat /tmp/pacificarium.pid)" 2>/dev/null || true
  rm -f /tmp/pacificarium.pid
fi
echo "Pacificarium stopped"
