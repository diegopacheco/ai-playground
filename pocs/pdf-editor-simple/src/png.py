import struct
import zlib


def encode(pixels, width, height, stride):
    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        start = y * stride
        raw.append(0)
        raw += pixels[start:start + row_bytes]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join([
        b"\x89PNG\r\n\x1a\n",
        _chunk(b"IHDR", header),
        _chunk(b"IDAT", zlib.compress(bytes(raw), 6)),
        _chunk(b"IEND", b""),
    ])


def _chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )
