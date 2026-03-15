import { describe, it, expect, beforeEach } from 'bun:test'
import { getState, setPlayerNames, setPlayerCards, startBattle, recordRound, resetGame, subscribe } from './store.js'

beforeEach(() => {
  resetGame()
})

describe('store', () => {
  it('should set player names', () => {
    setPlayerNames('Ash', 'Gary')
    const state = getState()
    expect(state.player1.name).toBe('Ash')
    expect(state.player2.name).toBe('Gary')
  })

  it('should set player cards', () => {
    const cards = [{ id: 1, name: 'pikachu', power: 200 }]
    setPlayerCards(1, cards)
    expect(getState().player1.cards).toEqual(cards)
  })

  it('should set player 2 cards', () => {
    const cards = [{ id: 4, name: 'charmander', power: 180 }]
    setPlayerCards(2, cards)
    expect(getState().player2.cards).toEqual(cards)
  })

  it('should start battle', () => {
    startBattle()
    const state = getState()
    expect(state.phase).toBe('battle')
    expect(state.currentRound).toBe(0)
  })

  it('should record a round with player 1 winning', () => {
    startBattle()
    const p1Card = { id: 1, name: 'pikachu', power: 300 }
    const p2Card = { id: 4, name: 'charmander', power: 200 }
    recordRound(p1Card, p2Card, 1)
    const state = getState()
    expect(state.player1.score).toBe(1)
    expect(state.currentRound).toBe(1)
    expect(state.rounds.length).toBe(1)
  })

  it('should record a round with player 2 winning', () => {
    startBattle()
    const p1Card = { id: 1, name: 'pikachu', power: 100 }
    const p2Card = { id: 4, name: 'charmander', power: 300 }
    recordRound(p1Card, p2Card, 2)
    expect(getState().player2.score).toBe(1)
  })

  it('should finish after 3 rounds and save to history', () => {
    setPlayerNames('Ash', 'Gary')
    startBattle()
    recordRound({ id: 1, name: 'a', power: 300 }, { id: 2, name: 'b', power: 200 }, 1)
    recordRound({ id: 3, name: 'c', power: 300 }, { id: 4, name: 'd', power: 200 }, 1)
    recordRound({ id: 5, name: 'e', power: 300 }, { id: 6, name: 'f', power: 200 }, 1)
    const state = getState()
    expect(state.phase).toBe('finished')
    expect(state.battleHistory.length).toBe(1)
    expect(state.battleHistory[0].champion).toBe('Ash')
  })

  it('should reset game', () => {
    startBattle()
    recordRound({ id: 1, name: 'a', power: 300 }, { id: 2, name: 'b', power: 200 }, 1)
    resetGame()
    const state = getState()
    expect(state.currentRound).toBe(0)
    expect(state.phase).toBe('setup')
    expect(state.player1.score).toBe(0)
  })

  it('should notify subscribers', () => {
    let notified = false
    const unsub = subscribe(() => { notified = true })
    setPlayerNames('A', 'B')
    expect(notified).toBe(true)
    unsub()
  })
})
