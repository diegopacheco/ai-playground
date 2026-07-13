#!/usr/bin/env bash
if [ -f /tmp/ybyra.pid ]; then
  kill "$(< /tmp/ybyra.pid)" 2>/dev/null || true
  rm -f /tmp/ybyra.pid
fi
echo "Ybyrá stopped"
