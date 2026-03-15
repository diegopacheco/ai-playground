let gameState = {
  player1: { name: 'Player 1', cards: [], score: 0 },
  player2: { name: 'Player 2', cards: [], score: 0 },
  currentRound: 0,
  rounds: [],
  battleHistory: [],
  phase: 'setup',
}

let listeners = new Set()

function notify() {
  listeners.forEach((fn) => fn(getState()))
}

export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function getState() {
  return { ...gameState }
}

export function setPlayerNames(p1, p2) {
  gameState.player1.name = p1
  gameState.player2.name = p2
  notify()
}

export function setPlayerCards(player, cards) {
  if (player === 1) gameState.player1.cards = cards
  else gameState.player2.cards = cards
  notify()
}

export function startBattle() {
  gameState.currentRound = 0
  gameState.rounds = []
  gameState.player1.score = 0
  gameState.player2.score = 0
  gameState.phase = 'battle'
  notify()
}

export function recordRound(p1Card, p2Card, winner) {
  gameState.rounds.push({ p1Card, p2Card, winner, round: gameState.currentRound + 1 })
  if (winner === 1) gameState.player1.score++
  else if (winner === 2) gameState.player2.score++
  gameState.currentRound++

  if (gameState.currentRound >= 3) {
    gameState.phase = 'finished'
    gameState.battleHistory.push({
      id: Date.now(),
      date: new Date().toLocaleString(),
      player1: { ...gameState.player1 },
      player2: { ...gameState.player2 },
      rounds: [...gameState.rounds],
      champion: gameState.player1.score > gameState.player2.score ? gameState.player1.name :
                gameState.player2.score > gameState.player1.score ? gameState.player2.name : 'Draw',
    })
  }
  notify()
}

export function resetGame() {
  gameState.currentRound = 0
  gameState.rounds = []
  gameState.player1.score = 0
  gameState.player2.score = 0
  gameState.player1.cards = []
  gameState.player2.cards = []
  gameState.phase = 'setup'
  notify()
}
