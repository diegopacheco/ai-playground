import { describe, it, expect } from "vitest"
import type { CardData, Game } from "./types"

function isGameComplete(game: Game): boolean {
  return game.matches_found === game.total_pairs
}

function getFlippedCards(board: CardData[]): CardData[] {
  return board.filter((c) => c.flipped && !c.matched)
}

function calculateScore(moves: number): number {
  return Math.max(1000 - moves * 10, 100)
}

function createBoard(pairs: number): CardData[] {
  const cards: CardData[] = []
  for (let v = 1; v <= pairs; v++) {
    cards.push({ position: cards.length, value: v, flipped: false, matched: false })
    cards.push({ position: cards.length, value: v, flipped: false, matched: false })
  }
  return cards
}

describe("game state logic", () => {
  it("detects game completion when all pairs matched", () => {
    const game: Game = {
      id: 1,
      player_id: 1,
      board: [],
      moves: 8,
      matches_found: 8,
      total_pairs: 8,
      status: "completed",
      score: 920,
    }
    expect(isGameComplete(game)).toBe(true)
  })

  it("detects game not complete when pairs remain", () => {
    const game: Game = {
      id: 1,
      player_id: 1,
      board: [],
      moves: 5,
      matches_found: 3,
      total_pairs: 8,
      status: "in_progress",
      score: 0,
    }
    expect(isGameComplete(game)).toBe(false)
  })

  it("returns flipped non-matched cards", () => {
    const board: CardData[] = [
      { position: 0, value: 1, flipped: true, matched: false },
      { position: 1, value: 2, flipped: false, matched: false },
      { position: 2, value: 1, flipped: false, matched: true },
      { position: 3, value: 2, flipped: true, matched: false },
    ]
    const flipped = getFlippedCards(board)
    expect(flipped).toHaveLength(2)
    expect(flipped[0].position).toBe(0)
    expect(flipped[1].position).toBe(3)
  })

  it("returns empty when no cards flipped", () => {
    const board: CardData[] = [
      { position: 0, value: 1, flipped: false, matched: false },
      { position: 1, value: 2, flipped: false, matched: false },
    ]
    expect(getFlippedCards(board)).toHaveLength(0)
  })

  it("excludes matched cards from flipped count", () => {
    const board: CardData[] = [
      { position: 0, value: 1, flipped: true, matched: true },
      { position: 1, value: 2, flipped: true, matched: false },
    ]
    const flipped = getFlippedCards(board)
    expect(flipped).toHaveLength(1)
    expect(flipped[0].position).toBe(1)
  })

  it("calculates score with perfect game", () => {
    expect(calculateScore(8)).toBe(920)
  })

  it("calculates score with many moves", () => {
    expect(calculateScore(50)).toBe(500)
  })

  it("calculates score minimum is 100", () => {
    expect(calculateScore(200)).toBe(100)
  })

  it("creates correct number of cards for board", () => {
    const board = createBoard(8)
    expect(board).toHaveLength(16)
  })

  it("creates correct pairs for board", () => {
    const board = createBoard(4)
    expect(board).toHaveLength(8)
    const counts: Record<number, number> = {}
    board.forEach((c) => {
      if (c.value !== null) {
        counts[c.value] = (counts[c.value] || 0) + 1
      }
    })
    expect(Object.keys(counts)).toHaveLength(4)
    Object.values(counts).forEach((count) => {
      expect(count).toBe(2)
    })
  })

  it("creates all cards face down", () => {
    const board = createBoard(8)
    board.forEach((card) => {
      expect(card.flipped).toBe(false)
      expect(card.matched).toBe(false)
    })
  })
})
