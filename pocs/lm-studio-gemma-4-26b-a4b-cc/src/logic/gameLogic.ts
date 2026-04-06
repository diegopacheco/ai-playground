export type Move = 'Rock' | 'Paper' | 'Scissors';
export type Result = 'Win' | 'Loss' | 'Draw';

export const getResult = (playerMove: Move, computerMove: Move): Result => {
  if (playerMove === computerMove) return 'Draw';
  if (
    (playerMove === 'Rock' && computerMove === 'Scissors') ||
    (playerMove === 'Paper' && computerMove === 'Rock') ||
    (playerMove === 'Scissors' && computerMove === 'Paper')
  ) {
    return 'Win';
  }
  return 'Loss';
};

export const getRandomMove = (): Move => {
  const moves: Move[] = ['Rock', 'Paper', 'Scissors'];
  return moves[Math.floor(Math.random() * moves.length)];
};
