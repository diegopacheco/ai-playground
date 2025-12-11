import { Choice } from '../types';

/**
 * Determines the winner of a rock paper scissors round
 * @param playerChoice - The player's choice
 * @param computerChoice - The computer's choice
 * @returns 'win' if player wins, 'lose' if computer wins, 'tie' if same choice
 */
export function determineWinner(playerChoice: Choice, computerChoice: Choice): 'win' | 'lose' | 'tie' {
  if (playerChoice === computerChoice) {
    return 'tie';
  }

  // Player wins scenarios
  if (
    (playerChoice === 'rock' && computerChoice === 'scissors') ||
    (playerChoice === 'scissors' && computerChoice === 'paper') ||
    (playerChoice === 'paper' && computerChoice === 'rock')
  ) {
    return 'win';
  }

  // Computer wins in all other cases
  return 'lose';
}

/**
 * Generates a random choice for the computer opponent
 * @returns A random choice from rock, paper, or scissors
 */
export function generateComputerChoice(): Choice {
  const choices: Choice[] = ['rock', 'paper', 'scissors'];
  const randomIndex = Math.floor(Math.random() * choices.length);
  return choices[randomIndex];
}