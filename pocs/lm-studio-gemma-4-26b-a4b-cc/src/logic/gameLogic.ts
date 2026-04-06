export type Move = 'Rock' | 'Paper' | 'Scissors'
export type Result = 'Win' | 'Loss' | 'Draw'

export interface GameResult {
  playerMove: Move
  computerMove: Move
  result: Result
  timestamp: number
}

export function getGameResult(playerMove: Move, computer/computerMove: Move): Result {
  if (playerMove === computerMove) return 'Draw'
  if (
    (playerMove === 'Rock' && computerMove === 'Scissors') ||
    (playerMove === 'Paper' && computerMove === 'Rock') ||
    (playerMove === 'Scissors' && computerMove === 'Paper')
  ) {
    return 'Win'
  }
  return 'Loss'
}
