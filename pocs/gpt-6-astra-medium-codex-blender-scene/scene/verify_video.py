import json
import subprocess
import sys
import wave
from pathlib import Path


folder = Path(sys.argv[1])
manifest = json.loads((folder / 'frostbite-falls.json').read_text())
video = folder / 'frostbite-falls.mp4'
probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(video)], check=True, capture_output=True, text=True)
data = json.loads(probe.stdout)
visual = next(stream for stream in data['streams'] if stream['codec_type'] == 'video')
audio = next(stream for stream in data['streams'] if stream['codec_type'] == 'audio')
assert visual['codec_name'] == 'h264'
assert visual['pix_fmt'] == 'yuv420p'
assert (visual['width'], visual['height']) == (manifest['width'], manifest['height'])
assert int(visual['nb_read_frames']) == manifest['frames'], 'Video must contain every animation frame'
assert abs(float(data['format']['duration']) - 16) < 0.1
assert audio['codec_name'] == 'aac'
with wave.open(str(folder / 'soundtrack.wav')) as source:
    assert source.getnframes() == 16 * source.getframerate()
    assert any(source.readframes(source.getnframes())), 'Soundtrack cannot be silent'
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
print(f"PASS: {visual['nb_read_frames']} frames, {visual['width']}x{visual['height']}, 16 seconds, H.264, AAC and full decode")
