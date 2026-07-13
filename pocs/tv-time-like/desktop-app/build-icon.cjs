const fs = require("node:fs")
const path = require("node:path")

const source = process.argv[2]
const output = process.argv[3]
const images = [
  ["icp4", "icon_16x16.png"],
  ["icp5", "icon_32x32.png"],
  ["icp6", "icon_32x32@2x.png"],
  ["ic07", "icon_128x128.png"],
  ["ic08", "icon_256x256.png"],
  ["ic09", "icon_512x512.png"],
  ["ic10", "icon_512x512@2x.png"]
]
const chunks = images.map(([type, file]) => {
  const image = fs.readFileSync(path.join(source, file))
  const header = Buffer.alloc(8)
  header.write(type, 0, 4, "ascii")
  header.writeUInt32BE(image.length + 8, 4)
  return Buffer.concat([header, image])
})
const body = Buffer.concat(chunks)
const header = Buffer.alloc(8)
header.write("icns", 0, 4, "ascii")
header.writeUInt32BE(body.length + 8, 4)
fs.writeFileSync(output, Buffer.concat([header, body]))
