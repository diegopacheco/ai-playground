import { test } from 'node:test'
import assert from 'node:assert/strict'
import { validateRequest, validateReply, validateCheats, maxMemoryCodes } from '../public/contracts.js'
import { buildPrompt, callCodex } from '../agent.mjs'
import { snesRam } from '../public/memory.js'
import { PatchEngine } from '../public/patch-engine.js'

const cheat = { label: 'Color', code: '7E00001F' }
const request = { prompt: 'Change color', game: { name: 'color.sfc', internalTitle: 'COLOR LAB', sha256: 'a'.repeat(64), size: 32768 }, cheats: [] }

test('only bounded WRAM writes reach the core', () => {
  for (const code of ['802000FF', '7E0000', '7E0000FF;alert(1)', '7E0000GG', null]) assert.throws(() => validateReply({ message: 'Apply', unsupported: false, cheats: [{ label: 'Color', code }] }))
  assert.throws(() => validateReply({ message: 'Apply', unsupported: false, cheats: [cheat, { label: 'Conflict', code: '7E000000' }] }))
  assert.throws(() => validateReply({ message: 'Cannot', unsupported: true, cheats: [cheat] }))
  assert.throws(() => validateReply({ message: 'Apply', unsupported: false, cheats: Array(maxMemoryCodes + 1).fill(cheat) }))
})

test('large memory batches preserve one-time semantics and still enforce a hard bound', () => {
  const cheats = Array.from({ length: 256 }, (_, index) => ({ label: 'Progress', code: (0x7e1000 + index).toString(16).toUpperCase() + '0F', once: true }))
  assert.deepEqual(validateCheats(cheats), cheats)
  assert.throws(() => validateCheats([...cheats, { label: 'Extra', code: '7E200000' }]), /at most 256/)
  assert.throws(() => validateCheats([{ ...cheat, once: 'true' }]), /boolean/)
  assert.match(buildPrompt(request), /At most 256/)
  assert.match(buildPrompt(request), /once=true/)
})

test('compact ranges expand generically without overlapping or exceeding work RAM', () => {
  assert.deepEqual(validateCheats([{ label: 'Flags', code: '7E10000F', count: 3, once: true, mask: 'DF' }]).map(cheat => cheat.code), ['7E10000F', '7E10010F', '7E10020F'])
  for (const count of [0, 257, 1.5]) assert.throws(() => validateCheats([{ label: 'Flags', code: '7E10000F', count }]), /ranges/)
  assert.throws(() => validateCheats([{ label: 'Flags', code: '7FFFFF0F', count: 2 }]), /ranges/)
  assert.throws(() => validateCheats([{ label: 'Flags', code: '7E10000F', count: 3 }, { label: 'Overlap', code: '7E100100' }]), /same address/)
})

test('one-time writes use game-independent RAM and do not occupy active slots', async () => {
  const ram = Buffer.alloc(131072)
  ram[0] = 10
  const state = new Uint8Array(Buffer.concat([Buffer.from('#!s9xsnp:0012\nRAM:131072:'), ram]))
  const core = emulator(state)
  const engine = new PatchEngine(core)
  await engine.apply([{ label: 'Progress', code: '7E000063', once: true }])
  assert.equal(snesRam(core.gameManager.getState())[0], 99)
  assert.deepEqual(engine.cheats, [])
  assert.deepEqual(core.codes, [])
  await engine.undo()
  assert.equal(snesRam(core.gameManager.getState())[0], 10)
})

test('access masks preserve earned progress and leave completion fields writable', async () => {
  const ram = Buffer.alloc(131072)
  ram[0x1000] = 0xe0
  ram[0x2000] = 0x84
  ram[0x2001] = 2
  const state = new Uint8Array(Buffer.concat([Buffer.from('#!s9xsnp:0012\nRAM:131072:'), ram]))
  const core = emulator(state)
  const engine = new PatchEngine(core)
  await engine.apply([{ label: 'Stage access', code: '7E10000F', once: true, mask: 'DF' }])
  const actual = snesRam(core.gameManager.getState())
  assert.equal(actual[0x1000], 0xcf)
  assert.deepEqual([...actual.subarray(0x2000, 0x2002)], [0x84, 2])
  assert.deepEqual(core.codes, [])
  actual[0x2000] |= 1
  actual[0x2001]++
  assert.deepEqual([...actual.subarray(0x2000, 0x2002)], [0x85, 3])
  await engine.undo()
  assert.equal(snesRam(core.gameManager.getState())[0x1000], 0xe0)
  assert.deepEqual([...snesRam(core.gameManager.getState()).subarray(0x2000, 0x2002)], [0x84, 2])
})

test('agent requests require cartridge identity and bounded intent', () => {
  assert.deepEqual(validateRequest(request), request)
  assert.equal(validateRequest({ ...request, model: 'gpt-5.6-sol' }).model, 'gpt-5.6-sol')
  assert.throws(() => validateRequest({ ...request, model: 'gpt-4o' }), /supported Codex model/)
  for (const value of [{ ...request, game: null }, { ...request, prompt: ' '.repeat(12) }, { ...request, prompt: 'a'.repeat(2001) }]) assert.throws(() => validateRequest(value))
  assert.match(buildPrompt(request), /Never invent addresses/)
})

test('every cartridge uses research without bundled game mappings', () => {
  const prompt = buildPrompt(request)
  assert.match(prompt, /Use web search/)
  assert.match(prompt, /There are no built-in game profiles/)
  assert.match(prompt, /Never copy one game's addresses into another game/)
  assert.match(prompt, /source URLs/)
  assert.match(prompt, /no access to live memory or ROM bytes/)
  assert.match(prompt, /Unlocking access must preserve earned progress/)
})

test('cancelled requests never reach the agent SDK', async () => {
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(() => callCodex(request, controller.signal), /Request cancelled/)
})

function emulator(initial = new Uint8Array([10, 20])) {
  let state = initial
  const codes = []
  return {
    paused: false,
    codes,
    pause() { this.paused = true },
    play() { this.paused = false },
    gameManager: {
      getState: () => state,
      loadState: value => { state = value.slice() },
      resetCheat: () => { codes.length = 0 },
      setCheat: (index, enabled, code) => { codes[index] = code }
    }
  }
}

test('undo restores pre-change gameplay and previous active codes', async () => {
  const core = emulator()
  const engine = new PatchEngine(core)
  await engine.apply([cheat])
  core.gameManager.getState()[0] = 99
  await engine.apply([{ label: 'Blue', code: '7E000000' }])
  core.gameManager.getState()[0] = 88
  await engine.undo()
  assert.equal(core.gameManager.getState()[0], 99)
  assert.deepEqual(core.codes, [cheat.code])
  await engine.undo()
  assert.deepEqual([...core.gameManager.getState()], [10, 20])
  assert.deepEqual(core.codes, [])
  assert.equal(core.paused, false)
})

test('apply and undo wait for a completed core pause before touching checkpoints', async () => {
  const core = emulator()
  const getState = core.gameManager.getState
  const loadState = core.gameManager.loadState
  let paused = false
  let pauses = 0
  core.gameManager.getState = () => { assert.equal(paused, true); return getState() }
  core.gameManager.loadState = state => { assert.equal(paused, true); loadState(state) }
  const engine = new PatchEngine(core, async () => {}, async () => {
    core.pause()
    await Promise.resolve()
    paused = true
    pauses++
  })
  await engine.apply([cheat])
  paused = false
  await engine.undo()
  assert.equal(pauses, 2)
  assert.equal(core.paused, false)
  assert.deepEqual([...getState()], [10, 20])
})

test('a failed checkpoint prevents mutation and preserves pause state', async () => {
  const core = emulator()
  core.paused = true
  core.gameManager.getState = () => new Uint8Array()
  const engine = new PatchEngine(core)
  await assert.rejects(() => engine.apply([cheat]), /checkpoint/)
  assert.deepEqual(core.codes, [])
  assert.equal(core.paused, true)
  assert.equal(engine.history.length, 0)
})

test('a failed code write rolls back state and active codes', async () => {
  const core = emulator()
  const engine = new PatchEngine(core)
  await engine.apply([cheat])
  core.gameManager.setCheat = (index, enabled, code) => {
    if (code === '7E000000') throw new Error('Core rejected write')
    core.codes[index] = code
  }
  await assert.rejects(() => engine.apply([{ label: 'Fail', code: '7E000000' }]), /rejected/)
  assert.deepEqual(core.codes, [cheat.code])
  assert.equal(engine.history.length, 1)
  assert.equal(core.paused, false)
})

test('checkpoints are capped to avoid unbounded save-state memory', async () => {
  const engine = new PatchEngine(emulator())
  for (let index = 0; index < 8; index++) await engine.apply([{ label: 'Color', code: `7E00000${index}` }])
  assert.equal(engine.history.length, 5)
})

test('undo drains queued cheat removal before restoring gameplay and keeps the pause state', async () => {
  const core = emulator()
  core.paused = true
  let queued = false
  const reset = core.gameManager.resetCheat
  core.gameManager.resetCheat = () => { reset(); queued = true }
  const engine = new PatchEngine(core, async () => { if (queued) core.gameManager.getState()[0] = 254; queued = false })
  await engine.apply([cheat])
  assert.equal(core.gameManager.getState()[0], 254)
  await engine.undo()
  assert.equal(core.gameManager.getState()[0], 10)
  assert.equal(core.paused, true)
})

test('a stalled core still restores checkpoint memory after verification cannot advance', async () => {
  const core = emulator()
  let first = true
  const engine = new PatchEngine(core, async () => {
    if (first) core.gameManager.getState()[0] = 99
    first = false
    throw new Error('Could not advance')
  })
  await assert.rejects(() => engine.apply([cheat]), /Could not advance/)
  assert.deepEqual([...core.gameManager.getState()], [10, 20])
  assert.equal(engine.history.length, 0)
  assert.equal(core.paused, false)
})
