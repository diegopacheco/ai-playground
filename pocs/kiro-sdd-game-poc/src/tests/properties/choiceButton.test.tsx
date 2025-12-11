import { describe, it, afterEach } from 'bun:test';
import * as fc from 'fast-check';
import { render, fireEvent, screen, cleanup } from '@testing-library/react';
import React from 'react';
import { ChoiceButton } from '../../components/ChoiceButton';
import { Choice } from '../../types';
import '../../test-setup';

// Arbitrary generator for valid choices
const choiceArbitrary = fc.constantFrom('rock', 'paper', 'scissors') as fc.Arbitrary<Choice>;

describe('ChoiceButton Properties', () => {
  afterEach(() => {
    cleanup();
  });
  /**
   * Feature: rock-paper-scissors-game, Property 1: Player choice registration
   * For any valid choice (rock, paper, scissors), when a player clicks the 
   * corresponding button, the game state should reflect that choice
   * Validates: Requirements 1.1
   */
  it('should register player choice when button is clicked', () => {
    fc.assert(
      fc.property(choiceArbitrary, (choice) => {
        try {
          let capturedChoice: Choice | null = null;
          
          // Mock onClick handler that captures the choice
          const handleClick = (selectedChoice: Choice) => {
            capturedChoice = selectedChoice;
          };

          // Render the ChoiceButton component with a container
          const { container, unmount } = render(
            <ChoiceButton
              choice={choice}
              onClick={handleClick}
              disabled={false}
            />
          );

          // Find and click the button within this specific container
          const button = container.querySelector(`[data-testid="choice-button-${choice}"]`) as HTMLElement;
          if (!button) {
            throw new Error(`Button not found for choice: ${choice}`);
          }
          
          fireEvent.click(button);

          // Clean up this render immediately
          unmount();

          // Verify that the correct choice was captured
          return capturedChoice === choice;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Disabled buttons should not register clicks
   * For any valid choice, when a button is disabled, clicking it should not 
   * trigger the onClick handler
   */
  it('should not register clicks when button is disabled', () => {
    fc.assert(
      fc.property(choiceArbitrary, (choice) => {
        try {
          let clickCount = 0;
          
          // Mock onClick handler that counts clicks
          const handleClick = () => {
            clickCount++;
          };

          // Render the ChoiceButton component as disabled
          const { container, unmount } = render(
            <ChoiceButton
              choice={choice}
              onClick={handleClick}
              disabled={true}
            />
          );

          // Find and click the button within this specific container
          const button = container.querySelector(`[data-testid="choice-button-${choice}"]`) as HTMLElement;
          if (!button) {
            throw new Error(`Button not found for choice: ${choice}`);
          }
          
          fireEvent.click(button);

          // Clean up this render immediately
          unmount();

          // Verify that no click was registered
          return clickCount === 0;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Button displays correct choice representation
   * For any valid choice, the button should display the correct emoji and text
   */
  it('should display correct choice representation', () => {
    fc.assert(
      fc.property(choiceArbitrary, (choice) => {
        try {
          const handleClick = () => {};

          // Render the ChoiceButton component
          const { container, unmount } = render(
            <ChoiceButton
              choice={choice}
              onClick={handleClick}
            />
          );

          // Find the button within this specific container
          const button = container.querySelector(`[data-testid="choice-button-${choice}"]`) as HTMLElement;
          if (!button) {
            throw new Error(`Button not found for choice: ${choice}`);
          }
          
          // Check that the button contains the expected text
          const expectedText = choice.charAt(0).toUpperCase() + choice.slice(1);
          const hasCorrectText = button.textContent?.includes(expectedText) || false;
          
          // Check that the button has the correct aria-label
          const expectedAriaLabel = `Select ${expectedText}`;
          const hasCorrectAriaLabel = button.getAttribute('aria-label') === expectedAriaLabel;
          
          // Clean up this render immediately
          unmount();
          
          return hasCorrectText && hasCorrectAriaLabel;
        } catch (error) {
          console.error('Property test error:', error);
          return false;
        }
      }),
      { numRuns: 100 }
    );
  });
});