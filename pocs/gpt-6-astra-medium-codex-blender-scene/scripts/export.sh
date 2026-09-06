#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require ffmpeg
require python3
[ -f "$BUILD/frostbite-falls.json" ] || fail 'Build the scene first'
python3 "$ROOT/scene/check_frames.py" "$BUILD"
[ -f "$BUILD/soundtrack.wav" ] || python3 "$ROOT/scene/soundtrack.py" "$BUILD/soundtrack.wav"
ffmpeg -hide_banner -loglevel warning -xerror -y -framerate "$fps" -start_number 1 -i "$BUILD/frames/%04d.png" -i "$BUILD/soundtrack.wav" -frames:v "$((fps * 16))" -c:v libx264 -preset medium -crf "${CRF:-23}" -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart "$BUILD/frostbite-falls.mp4"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate -of json "$BUILD/frostbite-falls.mp4"
