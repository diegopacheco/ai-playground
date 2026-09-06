import json
import struct
import sys
from pathlib import Path


def check(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'frostbite-falls.json').read_text())
    for frame in range(1, manifest['frames'] + 1):
        path = folder / 'frames' / f'{frame:04d}.png'
        if not path.is_file():
            raise ValueError(f'Missing frame {frame}: {path}')
        with path.open('rb') as source:
            header = source.read(24)
        if header[:8] != b'\x89PNG\r\n\x1a\n' or len(header) != 24:
            raise ValueError(f'Invalid PNG: {path}')
        if struct.unpack('>II', header[16:24]) != (manifest['width'], manifest['height']):
            raise ValueError(f'Wrong frame dimensions: {path}')
    return manifest


if __name__ == '__main__':
    result = check(sys.argv[1])
    print(f"Verified {result['frames']} consecutive PNG frames")
