#!/bin/bash
kill $(cat /tmp/megaslop-runner.pid 2>/dev/null) 2>/dev/null
lsof -ti:8088 | xargs kill 2>/dev/null
rm -f /tmp/megaslop-runner.pid
echo "MEGASLOP RUNNER stopped."
