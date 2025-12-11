import { describe, it } from 'bun:test';
import * as fc from 'fast-check';
import { determineWinner, generateComputerChoice } from '../../utils/gameLogic';
import { Choice } from '../../types';

// Arbitrary generator for valid choices
const choiceArbitrary = fc.constantFrom('rock', 'paper', 'scissors') as fc.Arbitrary<Choice>;

describe('Game Logic Properties', () => {
  /**
   * Feature: rock-paper-scissors-game, Property 3: Game logic correctness
   * For any combination of player and computer choices, the winner determination 
   * should follow standard rock paper scissors rules (rock beats scissors, 
   * scissors beats paper, paper beats rock, same choices tie)
   * Validates: Requirements 1.3
   */
  it('should follow standard rock paper scissors rules for all choice combinations', () => {
    fc.assert(
      fc.property(choiceArbitrary, choiceArbitrary, (playerChoice, computerChoice) => {
        const result = determineWinner(playerChoice, computerChoice);
        
        // Same choices should always tie
        if (playerChoice === computerChoice) {
          return result === 'tie';
        }
        
        // Player wins scenarios
        const playerWins = 
          (playerChoice === 'rock' && computerChoice === 'scissors') ||
          (playerChoice === 'scissors' && computerChoice === 'paper') ||
          (playerChoice === 'paper' && computerChoice === 'rock');
        
        if (playerWins) {
          return result === 'win';
        } else {
          return result === 'lose';
        }
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: rock-paper-scissors-game, Property 2: Computer choice generation
   * For any player choice, the system should generate a computer choice that is 
   * one of the three valid options (rock, paper, scissors)
   * Validates: Requirements 1.2
   */
  it('should generate valid computer choices', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1000 }), (_) => {
        const computerChoice = generateComputerChoice();
        const validChoices: Choice[] = ['rock', 'paper', 'scissors'];
        return validChoices.includes(computerChoice);
      }),
      { numRuns: 100 }
    );
  });
});