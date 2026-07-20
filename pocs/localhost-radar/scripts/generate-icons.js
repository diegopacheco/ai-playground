const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

const SUPERSAMPLE = 4;
const SCOPE = [11, 14, 12];
const GRID = [124, 190, 92];
const BEAM = [169, 255, 104];
const BLIP = [255, 184, 77];

const crcTable = Array.from({ length: 256 }, (_, number) => {
  let value = number;
  for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ value >>> 1 : value >>> 1;
  return value >>> 0;
});

const crc32 = buffer => {
  let crc = 0xffffffff;
  for (const byte of buffer) crc = crcTable[(crc ^ byte) & 255] ^ crc >>> 8;
  return (crc ^ 0xffffffff) >>> 0;
};

const chunk = (name, data) => {
  const type = Buffer.from(name);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([type, data])));
  return Buffer.concat([length, type, data, checksum]);
};

const render = dimension => {
  const pixels = Buffer.alloc(dimension * dimension * 4);
  const scale = dimension / 128;
  const set = (x, y, color) => {
    const px = Math.round(x);
    const py = Math.round(y);
    if (px < 0 || py < 0 || px >= dimension || py >= dimension) return;
    const offset = (py * dimension + px) * 4;
    pixels[offset] = color[0];
    pixels[offset + 1] = color[1];
    pixels[offset + 2] = color[2];
    pixels[offset + 3] = 255;
  };
  const disc = (cx, cy, radius, color) => {
    for (let y = cy - radius; y <= cy + radius; y++) {
      for (let x = cx - radius; x <= cx + radius; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) set(x, y, color);
      }
    }
  };
  const line = (x1, y1, x2, y2, width, color) => {
    const distance = Math.hypot(x2 - x1, y2 - y1);
    for (let point = 0; point <= distance; point += 0.35) {
      const ratio = point / distance;
      disc(x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio, width / 2, color);
    }
  };
  const ring = (cx, cy, radius, width, color) => {
    const step = 0.4 / Math.max(radius, 1) * 180 / Math.PI;
    for (let angle = 0; angle < 360; angle += step) {
      const radians = angle * Math.PI / 180;
      disc(cx + Math.cos(radians) * radius, cy + Math.sin(radians) * radius, width / 2, color);
    }
  };
  const wedge = (cx, cy, radius, from, to, color) => {
    for (let angle = from; angle <= to; angle += 0.2) {
      const radians = angle * Math.PI / 180;
      line(cx, cy, cx + Math.cos(radians) * radius, cy + Math.sin(radians) * radius, 1.5 * scale, color);
    }
  };
  const center = 64 * scale;
  disc(center, center, 62 * scale, SCOPE);
  wedge(center, center, 56 * scale, -46, -12, [58, 96, 40]);
  ring(center, center, 59 * scale, 7 * scale, BEAM);
  ring(center, center, 40 * scale, 3.5 * scale, GRID);
  ring(center, center, 20 * scale, 3.5 * scale, GRID);
  line(8 * scale, center, 120 * scale, center, 3.5 * scale, GRID);
  line(center, 8 * scale, center, 120 * scale, 3.5 * scale, GRID);
  line(center, center, 104 * scale, 24 * scale, 7 * scale, BEAM);
  disc(94 * scale, 46 * scale, 9 * scale, BLIP);
  return pixels;
};

const downsample = (source, size) => {
  const dimension = size * SUPERSAMPLE;
  const output = Buffer.alloc(size * size * 4);
  const samples = SUPERSAMPLE * SUPERSAMPLE;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let red = 0;
      let green = 0;
      let blue = 0;
      let alpha = 0;
      for (let sy = 0; sy < SUPERSAMPLE; sy++) {
        for (let sx = 0; sx < SUPERSAMPLE; sx++) {
          const offset = ((y * SUPERSAMPLE + sy) * dimension + x * SUPERSAMPLE + sx) * 4;
          const weight = source[offset + 3] / 255;
          red += source[offset] * weight;
          green += source[offset + 1] * weight;
          blue += source[offset + 2] * weight;
          alpha += weight;
        }
      }
      const target = (y * size + x) * 4;
      output[target] = alpha ? Math.round(red / alpha) : 0;
      output[target + 1] = alpha ? Math.round(green / alpha) : 0;
      output[target + 2] = alpha ? Math.round(blue / alpha) : 0;
      output[target + 3] = Math.round(alpha / samples * 255);
    }
  }
  return output;
};

const encode = (pixels, size) => {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y++) pixels.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8;
  header[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
};

for (const size of [16, 32, 48, 128]) {
  const pixels = downsample(render(size * SUPERSAMPLE), size);
  fs.writeFileSync(path.join(__dirname, "..", "assets", `icon-${size}.png`), encode(pixels, size));
}
