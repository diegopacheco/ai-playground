#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

started=1
if [ -f .server.pid ] && [ -f .server.port ]; then
  existing_pid="$(<.server.pid)"
  existing_port="$(<.server.port)"
  existing_command="$(ps -p "$existing_pid" -o command= 2>/dev/null || true)"
  if kill -0 "$existing_pid" 2>/dev/null && [[ "$existing_command" == *"-m http.server $existing_port"* ]]; then
    started=0
  fi
fi

./start.sh
port="$(<.server.port)"
page="$(curl -fsS "http://127.0.0.1:$port/index.html")"

case "$page" in
  *"DUST"*"DEPLOY"*) echo "HTML check passed on port $port" ;;
  *) echo "HTML check failed"; exit 1 ;;
esac

for asset in game.js style.css assets/vendor/three.module.min.js assets/vendor/three.core.min.js assets/vendor/GLTFLoader.js assets/utils/BufferGeometryUtils.js assets/models/blaster-n.glb assets/models/blaster-e.glb assets/models/blaster-p.glb assets/models/Textures/colormap.png assets/textures/concrete.jpg assets/textures/pavement.jpg; do
  curl -fsS "http://127.0.0.1:$port/$asset" >/dev/null
  echo "$asset passed"
done

if [ "$started" -eq 1 ]; then
  ./stop.sh
fi

echo "All checks passed"
