import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('check_frames', root / 'scene/check_frames.py')
frames = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frames)


class ExportGuardTests(unittest.TestCase):
    def test_missing_frame_prevents_shortened_export(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / 'frostbite-falls.json').write_text(json.dumps({'frames': 2, 'width': 640, 'height': 360}))
            (folder / 'frames').mkdir()
            (folder / 'frames/0001.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + struct.pack('>II', 640, 360))
            with self.assertRaisesRegex(ValueError, 'Missing frame 2'):
                frames.check(folder)

    def test_mixed_resolution_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / 'frostbite-falls.json').write_text(json.dumps({'frames': 1, 'width': 640, 'height': 360}))
            (folder / 'frames').mkdir()
            (folder / 'frames/0001.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + struct.pack('>II', 1280, 720))
            with self.assertRaisesRegex(ValueError, 'Wrong frame dimensions'):
                frames.check(folder)


if __name__ == '__main__':
    unittest.main()
