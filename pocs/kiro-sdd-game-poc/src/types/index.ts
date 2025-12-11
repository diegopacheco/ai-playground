export type Choice = 'rock' | 'paper' | 'scissors';

export interface GameState {
  playerChoice: Choice | null;
  computerChoice: Choice | null;
  gameResult: 'win' | 'lose' | 'tie' | null;
  isPlaying: boolean;
  animationState: 'idle' | 'selecting' | 'revealing' | 'showing-result';
}

export interface ScoreState {
  wins: number;
  losses: number;
  ties: number;
}