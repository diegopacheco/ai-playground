function qrMultiply(left, right) {
  let result = 0
  for (let index = 7; index >= 0; index -= 1) {
    result = (result << 1) ^ ((result >>> 7) * 0x11d)
    result ^= ((right >>> index) & 1) * left
  }
  return result
}

function qrDivisor(degree) {
  const result = new Array(degree).fill(0)
  result[degree - 1] = 1
  let root = 1
  for (let index = 0; index < degree; index += 1) {
    for (let item = 0; item < result.length; item += 1) {
      result[item] = qrMultiply(result[item], root)
      if (item + 1 < result.length) result[item] ^= result[item + 1]
    }
    root = qrMultiply(root, 2)
  }
  return result
}

function qrRemainder(data, divisor) {
  const result = new Array(divisor.length).fill(0)
  for (const byte of data) {
    const factor = byte ^ result.shift()
    result.push(0)
    for (let index = 0; index < result.length; index += 1) result[index] ^= qrMultiply(divisor[index], factor)
  }
  return result
}

function qrCodewords(text) {
  const bytes = Array.from(new TextEncoder().encode(text))
  if (bytes.length > 53) throw new Error('Pairing address is too long')
  const bits = []
  const append = (value, count) => {
    for (let shift = count - 1; shift >= 0; shift -= 1) bits.push((value >>> shift) & 1)
  }
  append(4, 4)
  append(bytes.length, 8)
  for (const byte of bytes) append(byte, 8)
  for (let index = 0; index < Math.min(4, 440 - bits.length); index += 1) bits.push(0)
  while (bits.length % 8) bits.push(0)
  const data = []
  for (let index = 0; index < bits.length; index += 8) {
    let byte = 0
    for (let shift = 0; shift < 8; shift += 1) byte = (byte << 1) | bits[index + shift]
    data.push(byte)
  }
  for (let pad = 0; data.length < 55; pad += 1) data.push(pad % 2 === 0 ? 0xec : 0x11)
  return data.concat(qrRemainder(data, qrDivisor(15)))
}

function createQr(text) {
  const size = 29
  const modules = Array.from({ length: size }, () => new Array(size).fill(false))
  const functions = Array.from({ length: size }, () => new Array(size).fill(false))
  const setFunction = (x, y, dark) => {
    if (x < 0 || y < 0 || x >= size || y >= size) return
    modules[y][x] = dark
    functions[y][x] = true
  }
  const finder = (cx, cy) => {
    for (let dy = -4; dy <= 4; dy += 1) {
      for (let dx = -4; dx <= 4; dx += 1) {
        const distance = Math.max(Math.abs(dx), Math.abs(dy))
        setFunction(cx + dx, cy + dy, distance !== 2 && distance !== 4)
      }
    }
  }
  const alignment = (cx, cy) => {
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) setFunction(cx + dx, cy + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1)
    }
  }
  const format = (mask) => {
    const data = (1 << 3) | mask
    let remainder = data
    for (let index = 0; index < 10; index += 1) remainder = (remainder << 1) ^ ((remainder >>> 9) * 0x537)
    const bits = ((data << 10) | remainder) ^ 0x5412
    const bit = (index) => ((bits >>> index) & 1) !== 0
    for (let index = 0; index <= 5; index += 1) setFunction(8, index, bit(index))
    setFunction(8, 7, bit(6))
    setFunction(8, 8, bit(7))
    setFunction(7, 8, bit(8))
    for (let index = 9; index < 15; index += 1) setFunction(14 - index, 8, bit(index))
    for (let index = 0; index < 8; index += 1) setFunction(size - 1 - index, 8, bit(index))
    for (let index = 8; index < 15; index += 1) setFunction(8, size - 15 + index, bit(index))
    setFunction(8, size - 8, true)
  }

  for (let index = 0; index < size; index += 1) {
    setFunction(6, index, index % 2 === 0)
    setFunction(index, 6, index % 2 === 0)
  }
  finder(3, 3)
  finder(size - 4, 3)
  finder(3, size - 4)
  alignment(22, 22)
  format(0)

  const codewords = qrCodewords(text)
  let bitIndex = 0
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5
    for (let vertical = 0; vertical < size; vertical += 1) {
      const upward = ((right + 1) & 2) === 0
      const y = upward ? size - 1 - vertical : vertical
      for (let column = 0; column < 2; column += 1) {
        const x = right - column
        if (functions[y][x]) continue
        const dark = bitIndex < codewords.length * 8 && ((codewords[bitIndex >>> 3] >>> (7 - (bitIndex & 7))) & 1) !== 0
        modules[y][x] = dark !== ((x + y) % 2 === 0)
        bitIndex += 1
      }
    }
  }
  format(0)
  return modules
}

window.renderPairingQr = (canvas, text) => {
  const modules = createQr(text)
  const quiet = 4
  const scale = 6
  const size = (modules.length + quiet * 2) * scale
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d')
  context.fillStyle = '#fff0c9'
  context.fillRect(0, 0, size, size)
  context.fillStyle = '#17131e'
  for (let y = 0; y < modules.length; y += 1) {
    for (let x = 0; x < modules.length; x += 1) {
      if (modules[y][x]) context.fillRect((x + quiet) * scale, (y + quiet) * scale, scale, scale)
    }
  }
}
