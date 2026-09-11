export const models = ['gpt-6-astra', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna']
export const defaultModel = models[0]
export const maxMemoryCodes = 256
export const maxRequestBytes = 65536

export function validateCheats(cheats) {
  if (!Array.isArray(cheats) || cheats.length > maxMemoryCodes) throw new Error(`Use at most ${maxMemoryCodes} memory codes.`)
  const codes = new Set()
  return cheats.flatMap(cheat => {
    if (!cheat || typeof cheat.label !== 'string' || !cheat.label.trim() || cheat.label.length > 100 || typeof cheat.code !== 'string' || !/^7[EF][0-9A-F]{6}$/i.test(cheat.code)) throw new Error('Each change needs a label and an eight-digit SNES WRAM code beginning with 7E or 7F.')
    const code = cheat.code.toUpperCase()
    if (cheat.once !== undefined && typeof cheat.once !== 'boolean') throw new Error('A one-time memory write must use a boolean once flag.')
    if (cheat.mask != null && (!cheat.once || typeof cheat.mask !== 'string' || !/^[0-9A-F]{2}$/i.test(cheat.mask))) throw new Error('A preservation mask needs two hex digits and a one-time write.')
    const count = cheat.count ?? 1
    const address = parseInt(code.slice(0, 6), 16)
    if (!Number.isInteger(count) || count < 1 || count > maxMemoryCodes || address + count - 1 > 0x7fffff || codes.size + count > maxMemoryCodes) throw new Error(`Memory ranges must fit within WRAM and ${maxMemoryCodes} total bytes.`)
    return Array.from({ length: count }, (_, offset) => {
      const target = (address + offset).toString(16).toUpperCase()
      if (codes.has(target)) throw new Error('Two changes cannot write to the same address.')
      codes.add(target)
      return { label: cheat.label.trim(), code: target + code.slice(6), ...(cheat.once ? { once: true } : {}), ...(cheat.mask != null ? { mask: cheat.mask.toUpperCase() } : {}) }
    })
  })
}

export function validateRequest(body) {
  if (!body || typeof body.prompt !== 'string' || !body.prompt.trim() || body.prompt.length > 2000) throw new Error('Enter a request between 1 and 2,000 characters.')
  const game = body.game
  if (!game || typeof game.name !== 'string' || !game.name.trim() || game.name.length > 200 || typeof game.internalTitle !== 'string' || game.internalTitle.length > 21 || !/^[a-f0-9]{64}$/i.test(game.sha256) || !Number.isInteger(game.size) || game.size < 1 || game.size > 64 * 1024 * 1024) throw new Error('Load a valid SNES cartridge first.')
  const request = { prompt: body.prompt.trim(), game: { name: game.name, internalTitle: game.internalTitle, sha256: game.sha256, size: game.size }, cheats: validateCheats(body.cheats) }
  if (body.model !== undefined) {
    if (typeof body.model !== 'string' || !models.includes(body.model)) throw new Error('Choose a supported Codex model.')
    request.model = body.model
  }
  return request
}

export function validateReply(reply) {
  if (!reply || typeof reply.message !== 'string' || !reply.message.trim() || reply.message.length > 2000 || typeof reply.unsupported !== 'boolean') throw new Error('Codex returned an invalid change plan.')
  const cheats = validateCheats(reply.cheats)
  if (reply.unsupported && cheats.length) throw new Error('An unsupported request cannot contain changes.')
  return { message: reply.message.trim(), unsupported: reply.unsupported, cheats }
}
