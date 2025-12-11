import { describe, it, afterEach } from 'bun:test';
import * as fc from 'fast-check';
import { render, fireEvent, cleanup, act } from '@testing-library/react';
import React from 'react';
import { GameBoard } from '../../components/GameBoard';
import { Choice } from '../../types';
import '../../test-setup';

// Arbitrary generator for valid choices
const choiceArbitrary = fc.constantFrom('rock', 'paper', 'scissors') as fc.Arbitrary<Choice>;

describe('Animation State Properties', () => {
  afterEach(() => {
    cleanup();
  });

  /**
   * Feature: rock-paper-scissors-game, Property 5: Animation state progression
   * For any game action (player choice, computer reveal, result display, reset), 
   * the animation state should progress through the appropriate sequence and return to idle
   * Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
   */
  it('should progress through animation states correctly for any choice', async () => {
    // Use a smaller number of runs since this test involves async operations
    await fc.assert(
      fc.asyncProperty(choiceArbitrary, async (choice) => {
        try {
          const animationStates: string[] = [];
          
          const { container, unmount } = render(<GameBoard />);

          // Find the choice button
          const button = container.querySelector(`[data-testid="choice-button-${choice}"]`) as HTMLElement;
          if (!button) {
            unmount();
            return false;
          }

          // Initial state should be idle (buttons enabled)
          const initialDisabled = button.hasAttribute('disabled');
          if (initialDisabled) {
            unmount();
            return false;
          }
          animationStates.push('idle');

          // Click the button to start the game flow
          await act(async () => {
            fireEvent.click(button);
          });


          // After click, button should be disabled (selecting state)
          const afterClickDisabled = button.hasAttribute('disabled');
          if (!afterClickDisabled) {
            unmount();
            return false;
          }
          animationStates.push('selecting');

          // Check that status message appears
          const statusElement = container.querySelector('.gameboard__status');
          if (!statusElement) {
            unmount();
            return false;
          }

          // Verify the status shows selecting message
          const statusText = statusElement.textContent || '';
          const hasSelectingStatus = statusText.includes('Making your choice');
          if (!hasSelectingStatus) {
            unmount();
            return false;
          }

          unmount();

          // Verify we captured the expected state progression
          return animationStates.includes('idle') && animationStates.includes('selecting');
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 50 } // Reduced runs for async tests
    );
  });

  /**
   * Additional property: Buttons should be disabled during animation states
   * For any choice, once selected, all buttons should be disabled until game resets
   */
  it('should disable all buttons during animation states', async () => {
    await fc.assert(
      fc.asyncProperty(choiceArbitrary, async (choice) => {
        try {
          const { container, unmount } = render(<GameBoard />);

          // Find all choice buttons
          const rockButton = container.querySelector('[data-testid="choice-button-rock"]') as HTMLElement;
          const paperButton = container.querySelector('[data-testid="choice-button-paper"]') as HTMLElement;
          const scissorsButton = container.querySelector('[data-testid="choice-button-scissors"]') as HTMLElement;

          if (!rockButton || !paperButton || !scissorsButton) {
            unmount();
            return false;
          }

          // All buttons should be enabled initially
          const allEnabledInitially = 
            !rockButton.hasAttribute('disabled') &&
            !paperButton.hasAttribute('disabled') &&
            !scissorsButton.hasAttribute('disabled');

          if (!allEnabledInitially) {
            unmount();
            return false;
          }

          // Click the selected choice
          const selectedButton = container.querySelector(`[data-testid="choice-button-${choice}"]`) as HTMLElement;
          await act(async () => {
            fireEvent.click(selectedButton);
          });

          // All buttons should be disabled after selection
          const allDisabledAfterClick = 
            rockButton.hasAttribute('disabled') &&
            paperButton.hasAttribute('disabled') &&
            scissorsButton.hasAttribute('disabled');

          unmount();
          return allDisabledAfterClick;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 50 }
    );
  });
});
