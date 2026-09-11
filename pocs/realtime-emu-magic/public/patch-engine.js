import { snesRam } from './memory.js'
import { validateCheats } from './contracts.js'

export class PatchEngine {
  constructor(emulator, settle = async () => {}, pause = async () => emulator.pause()) {
    this.emulator = emulator
    this.settle = settle
    this.pause = pause
    this.cheats = []
    this.history = []
  }

  write(cheats) {
    const manager = this.emulator.gameManager
    manager.resetCheat()
    cheats.forEach((cheat, index) => manager.setCheat(index, true, cheat.code))
  }

  async transaction(action) {
    const wasPaused = this.emulator.paused
    try {
      await this.pause()
      return await action()
    } finally {
      if (!wasPaused) this.emulator.play()
    }
  }

  async restore(checkpoint) {
    try {
      this.write(checkpoint.cheats)
      await this.settle()
    } finally {
      this.emulator.gameManager.loadState(checkpoint.state)
      await this.settle()
    }
  }

  async apply(value) {
    const requested = validateCheats(value)
    return this.transaction(async () => {
      const state = this.emulator.gameManager.getState()
      if (!(state instanceof Uint8Array) || !state.length) throw new Error('Could not create a checkpoint. No changes applied.')
      const checkpoint = { state: state.slice(), cheats: this.cheats }
      const cheats = requested
      let verification
      try {
        const once = cheats.filter(cheat => cheat.once)
        let expectedRam
        if (once.length) {
          const patched = state.slice()
          const ram = snesRam(patched)
          for (const cheat of once) {
            const address = parseInt(cheat.code.slice(0, 6), 16) - 0x7e0000
            ram[address] = (ram[address] & parseInt(cheat.mask || '00', 16)) | parseInt(cheat.code.slice(6), 16)
          }
          expectedRam = ram.slice()
          await this.restore({ state: patched, cheats: cheats.filter(cheat => !cheat.once) })
        } else this.write(cheats)
        await this.settle()
        if (once.length) {
          const actual = snesRam(this.emulator.gameManager.getState())
          for (const cheat of once) {
            const address = parseInt(cheat.code.slice(0, 6), 16) - 0x7e0000
            if (actual[address] !== expectedRam[address]) throw new Error(`The emulator did not retain ${cheat.label}. Restored the checkpoint.`)
          }
          verification = `Verified ${once.length} one-time memory writes. Undo restores the checkpoint.`
        }
      } catch (error) {
        await this.restore(checkpoint)
        throw error
      }
      this.history.push(checkpoint)
      if (this.history.length > 5) this.history.shift()
      this.cheats = cheats.filter(cheat => !cheat.once)
      return { ...this.status(), verification }
    })
  }

  async undo() {
    const checkpoint = this.history.at(-1)
    if (!checkpoint) throw new Error('There is no checkpoint to restore.')
    return this.transaction(async () => {
      await this.restore(checkpoint)
      this.cheats = checkpoint.cheats
      this.history.pop()
      return this.status()
    })
  }

  status() {
    return { cheats: this.cheats, checkpoints: this.history.length }
  }
}
