export type Feedback = "HOT" | "COLD";

export type GuessResult = "CORRECT" | "WRONG";

export type MoveType = "question" | "guess";

export interface Turn {
  type: MoveType;
  text: string;
  reasoning: string;
  response: Feedback | GuessResult | null;
}

export interface Move {
  type: MoveType;
  text: string;
  reasoning: string;
}

export type GameResult = "win" | "loss";

export interface GameRecord {
  id: string;
  secret: string;
  turns: Turn[];
  result: GameResult;
  guessesUsed: number;
  createdAt: string;
  finishedAt: string;
}

export const MAX_GUESSES = 5;
export const MAX_QUESTIONS = 10;
