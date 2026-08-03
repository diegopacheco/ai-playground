const fs = require('node:fs')
const vm = require('node:vm')

const source = fs.readFileSync('public/qr.js', 'utf8')
const context = { TextEncoder, window: {} }
vm.createContext(context)
vm.runInContext(source, context)

const uri = 'http://192.168.1.20:8080/pair'
const codewords = context.qrCodewords(uri)
const matrix = context.createQr(uri)

if (codewords.length !== 70) process.exit(1)
if (matrix.length !== 29 || matrix.some((row) => row.length !== 29)) process.exit(1)
if (!context.qrRemainder(codewords, context.qrDivisor(15)).every((byte) => byte === 0)) process.exit(1)

for (const [x, y] of [[0, 0], [6, 0], [0, 6], [22, 0], [28, 6], [0, 22], [6, 28], [22, 22]]) {
  if (!matrix[y][x]) process.exit(1)
}

let painted = 0
const canvas = {
  width: 0,
  height: 0,
  getContext: () => ({
    fillStyle: '',
    fillRect: () => { painted += 1 }
  })
}
context.window.renderPairingQr(canvas, uri)
if (canvas.width !== 222 || canvas.height !== 222 || painted < 250) process.exit(1)
process.stdout.write('Local QR structure passed\n')
