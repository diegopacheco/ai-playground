#!/usr/bin/env bash
set -euo pipefail

input="$1"
tmp="${input%.mp4}.min.mp4"

ffmpeg -y -i "$input" -an -c:v libx264 -crf 32 -preset veryslow \
  -pix_fmt yuv420p -movflags +faststart \
  -vf "scale='min(1280,iw)':-2" "$tmp" >/dev/null 2>&1

before=$(wc -c < "$input" | tr -d ' ')
after=$(wc -c < "$tmp" | tr -d ' ')

if [ "$after" -lt "$before" ]; then
  mv "$tmp" "$input"
  echo "reduced $input $before -> $after bytes"
else
  rm -f "$tmp"
  echo "kept $input $before bytes (reduction not smaller)"
fi
