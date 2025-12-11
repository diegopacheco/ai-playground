import { describe, it, afterEach } from 'bun:test';
import * as fc from 'fast-check';
import { render, cleanup } from '@testing-library/react';
import React from 'react';
import { ResultDisplay } from '../../components/ResultDisplay';
import { Choice } from '../../types';
import '../../test-setup';

// Arbitrary generator for valid choices
const choiceArbitrary = fc.constantFrom('rock', 'paper', 'scissors') as fc.Arbitrary<Choice>;

// Arbitrary generator for game results
const gameResultArbitrary = fc.constantFrom('win', 'lose', 'tie') as fc.Arbitrary<'win' | 'lose' | 'tie'>;

// Helper to get expected emoji for a choice
const getExpectedEmoji = (choice: Choice): string => {
  switch (choice) {
    case 'rock': return '🪨';
    case 'paper': return '📄';
    case 'scissors': return '✂️';
  }
};

// Helper to get expected display name for a choice
const getExpectedDisplayName = (choice: Choice): string => {
  return choice.charAt(0).toUpperCase() + choice.slice(1);
};

describe('ResultDisplay Properties', () => {
  afterEach(() => {
    cleanup();
  });

  /**
   * Feature: rock-paper-scissors-game, Property 4: Choice display completeness
   * For any completed game round, both player and computer choices should be 
   * visually displayed with appropriate representations
   * Validates: Requirements 1.4, 1.5
   */
  it('should display both choices with correct visual representations', () => {
    fc.assert(
      fc.property(choiceArbitrary, choiceArbitrary, gameResultArbitrary, (playerChoice, computerChoice, gameResult) => {
        try {
          const { container, unmount } = render(
            <ResultDisplay
              playerChoice={playerChoice}
              computerChoice={computerChoice}
              gameResult={gameResult}
              isVisible={true}
            />
          );

          // Check player choice display
          const playerDisplay = container.querySelector('[data-testid="player-choice-display"]');
          if (!playerDisplay) {
            unmount();
            return false;
          }

          const playerHasEmoji = playerDisplay.textContent?.includes(getExpectedEmoji(playerChoice)) || false;
          const playerHasName = playerDisplay.textContent?.includes(getExpectedDisplayName(playerChoice)) || false;

          // Check computer choice display
          const computerDisplay = container.querySelector('[data-testid="computer-choice-display"]');
          if (!computerDisplay) {
            unmount();
            return false;
          }
          
          const computerHasEmoji = computerDisplay.textContent?.includes(getExpectedEmoji(computerChoice)) || false;
          const computerHasName = computerDisplay.textContent?.includes(getExpectedDisplayName(computerChoice)) || false;

          unmount();

          return playerHasEmoji && playerHasName && computerHasEmoji && computerHasName;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: rock-paper-scissors-game, Property 8: Result message accuracy
   * For any game outcome (win, lose, tie), the displayed message should 
   * correctly correspond to the actual result
   * Validates: Requirements 5.1, 5.2, 5.3
   */
  it('should display correct result message for any outcome', () => {
    fc.assert(
      fc.property(choiceArbitrary, choiceArbitrary, gameResultArbitrary, (playerChoice, computerChoice, gameResult) => {
        try {
          const { container, unmount } = render(
            <ResultDisplay
              playerChoice={playerChoice}
              computerChoice={computerChoice}
              gameResult={gameResult}
              isVisible={true}
            />
          );

          const resultMessage = container.querySelector('[data-testid="result-message"]');
          if (!resultMessage) {
            unmount();
            return false;
          }

          const messageText = resultMessage.textContent || '';
          let isCorrectMessage = false;

          switch (gameResult) {
            case 'win':
              isCorrectMessage = messageText.includes('You Win!');
              break;
            case 'lose':
              isCorrectMessage = messageText.includes('Computer Wins!');
              break;
            case 'tie':
              isCorrectMessage = messageText.includes("It's a Tie!");
              break;
          }

          unmount();
          return isCorrectMessage;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });


  /**
   * Feature: rock-paper-scissors-game, Property 9: Visual result highlighting
   * For any completed round, the winning choice should have distinct visual 
   * styling compared to the losing choice
   * Validates: Requirements 5.4, 5.5
   */
  it('should highlight winning choice with distinct styling', () => {
    fc.assert(
      fc.property(choiceArbitrary, choiceArbitrary, gameResultArbitrary, (playerChoice, computerChoice, gameResult) => {
        try {
          const { container, unmount } = render(
            <ResultDisplay
              playerChoice={playerChoice}
              computerChoice={computerChoice}
              gameResult={gameResult}
              isVisible={true}
            />
          );

          const playerDisplay = container.querySelector('[data-testid="player-choice-display"]');
          const computerDisplay = container.querySelector('[data-testid="computer-choice-display"]');

          if (!playerDisplay || !computerDisplay) {
            unmount();
            return false;
          }

          let isCorrectHighlighting = false;

          switch (gameResult) {
            case 'win':
              // Player should be winner, computer should be loser
              isCorrectHighlighting = 
                playerDisplay.classList.contains('result-display__choice--winner') &&
                computerDisplay.classList.contains('result-display__choice--loser');
              break;
            case 'lose':
              // Computer should be winner, player should be loser
              isCorrectHighlighting = 
                playerDisplay.classList.contains('result-display__choice--loser') &&
                computerDisplay.classList.contains('result-display__choice--winner');
              break;
            case 'tie':
              // Neither should have winner or loser class
              isCorrectHighlighting = 
                !playerDisplay.classList.contains('result-display__choice--winner') &&
                !playerDisplay.classList.contains('result-display__choice--loser') &&
                !computerDisplay.classList.contains('result-display__choice--winner') &&
                !computerDisplay.classList.contains('result-display__choice--loser');
              break;
          }

          unmount();
          return isCorrectHighlighting;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });
});
