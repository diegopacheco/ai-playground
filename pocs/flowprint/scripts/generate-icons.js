const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

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

const create = size => {
  const pixels = Buffer.alloc(size * size * 4);
  const scale = size / 128;
  const set = (x, y, color) => {
    const px = Math.round(x);
    const py = Math.round(y);
    if (px < 0 || py < 0 || px >= size || py >= size) return;
    const offset = (py * size + px) * 4;
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
  const arc = (cx, cy, radius, start, end, width, color) => {
    for (let angle = start; angle <= end; angle += 1 / Math.max(scale, .25)) {
      const radians = angle * Math.PI / 180;
      disc(cx + Math.cos(radians) * radius, cy + Math.sin(radians) * radius, width / 2, color);
    }
  };
  const paper = [242, 239, 230];
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) set(x, y, paper);
  const red = [239, 62, 47];
  arc(64 * scale, 67 * scale, 45 * scale, 155, 505, 8 * scale, red);
  arc(64 * scale, 69 * scale, 30 * scale, 150, 465, 8 * scale, red);
  arc(64 * scale, 72 * scale, 16 * scale, 145, 420, 7 * scale, red);
  disc(68 * scale, 87 * scale, 7 * scale, [45, 91, 255]);
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y++) pixels.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8;
  header[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
};

for (const size of [16, 32, 48, 128]) fs.writeFileSync(path.join(__dirname, "..", "assets", `icon-${size}.png`), create(size));
