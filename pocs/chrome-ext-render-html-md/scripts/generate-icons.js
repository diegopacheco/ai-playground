const fs = require("node:fs")
const path = require("node:path")
const zlib = require("node:zlib")

const output = process.argv[2]
const sizes = [16, 32, 48, 128]

function crc32(buffer) {
  let crc = 0xffffffff
  for (const byte of buffer) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
  }
  return (crc ^ 0xffffffff) >>> 0
}

function chunk(type, data) {
  const name = Buffer.from(type)
  const length = Buffer.alloc(4)
  const checksum = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  checksum.writeUInt32BE(crc32(Buffer.concat([name, data])))
  return Buffer.concat([length, name, data, checksum])
}

function colorAt(x, y, size) {
  const unitX = x / size * 128
  const unitY = y / size * 128
  const corner = 24
  const outside =
    unitX < corner && unitY < corner && (unitX - corner) ** 2 + (unitY - corner) ** 2 > corner ** 2 ||
    unitX > 128 - corner && unitY < corner && (unitX - 128 + corner) ** 2 + (unitY - corner) ** 2 > corner ** 2 ||
    unitX < corner && unitY > 128 - corner && (unitX - corner) ** 2 + (unitY - 128 + corner) ** 2 > corner ** 2 ||
    unitX > 128 - corner && unitY > 128 - corner && (unitX - 128 + corner) ** 2 + (unitY - 128 + corner) ** 2 > corner ** 2

  if (outside) return [0, 0, 0, 0]

  let color = [36, 35, 31, 255]
  const paper = unitX >= 29 && unitX <= 99 && unitY >= 18 && unitY <= 110 && !(unitX > 81 && unitY < 36 && unitX - 81 > unitY - 18)
  const fold = unitX >= 81 && unitX <= 99 && unitY >= 18 && unitY <= 37 && unitX - 81 <= unitY - 18

  if (paper) color = [255, 253, 247, 255]
  if (fold) color = [217, 210, 195, 255]

  const line = unitX >= 45 && unitX <= 83 && (unitY >= 55 && unitY <= 62 || unitY >= 69 && unitY <= 76) || unitX >= 45 && unitX <= 70 && unitY >= 83 && unitY <= 90
  if (line) color = [189, 60, 31, 255]
  return color
}

function createIcon(size) {
  const rows = []
  for (let y = 0; y < size; y += 1) {
    const row = Buffer.alloc(1 + size * 4)
    for (let x = 0; x < size; x += 1) {
      const color = colorAt(x + 0.5, y + 0.5, size)
      color.forEach((channel, index) => row.writeUInt8(channel, 1 + x * 4 + index))
    }
    rows.push(row)
  }

  const header = Buffer.alloc(13)
  header.writeUInt32BE(size, 0)
  header.writeUInt32BE(size, 4)
  header.writeUInt8(8, 8)
  header.writeUInt8(6, 9)
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])
  return Buffer.concat([
    signature,
    chunk("IHDR", header),
    chunk("IDAT", zlib.deflateSync(Buffer.concat(rows))),
    chunk("IEND", Buffer.alloc(0))
  ])
}

fs.mkdirSync(output, { recursive: true })
for (const size of sizes) fs.writeFileSync(path.join(output, `icon${size}.png`), createIcon(size))
