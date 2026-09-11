export function snesRam(state) {
  const text = new TextDecoder()
  let offset = 0
  if (text.decode(state.subarray(0, 7)) === 'RASTATE') {
    offset = 8
    const view = new DataView(state.buffer, state.byteOffset, state.byteLength)
    while (offset + 8 <= state.length) {
      const tag = text.decode(state.subarray(offset, offset + 4))
      const length = view.getUint32(offset + 4, true)
      if (offset + 8 + length > state.length) break
      offset += 8
      if (tag === 'MEM ') break
      offset += (length + 7) & ~7
    }
  }
  if (!/^#!s9xsnp:\d{4}\n$/.test(text.decode(state.subarray(offset, offset + 14)))) throw new Error('Could not read SNES memory for verification.')
  offset += 14
  while (offset + 11 <= state.length) {
    const header = text.decode(state.subarray(offset, offset + 11))
    if (!/^[A-Z0-9]{3}:\d{6}:$/.test(header)) break
    const length = Number(header.slice(4, 10))
    offset += 11
    if (offset + length > state.length) break
    if (header.startsWith('RAM:') && length === 131072) return state.subarray(offset, offset + length)
    offset += length
  }
  throw new Error('The save state has no readable SNES work RAM.')
}

