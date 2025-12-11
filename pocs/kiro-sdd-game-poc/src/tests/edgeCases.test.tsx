import React from 'react';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import '@testing-library/jest-dom';
import { StorageService } from '../services/StorageService';
import { ScoreBoard } from '../components/ScoreBoard';
import { GameBoard } from '../components/GameBoard';
import { ChoiceButton } from '../components/ChoiceButton';

// Create localStorage mock
const createLocalStorageMock = () => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] || null
  };
};

/**
 * Edge case tests for localStorage corruption scenarios
 * Validates: Requirements 3.3
 */
describe('localStorage Corruption Edge Cases', () => {
  let mockLocalStorage: ReturnType<typeof createLocalStorageMock>;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    originalLocalStorage = global.localStorage;
    mockLocalStorage = createLocalStorageMock();
    (global as any).localStorage = mockLocalStorage;
  });

  afterEach(() => {
    (global as any).localStorage = originalLocalStorage;
    cleanup();
  });

  it('should handle corrupted JSON in localStorage gracefully', () => {
    // Set corrupted JSON
    mockLocalStorage.setItem('rps-scores', 'not valid json {{{');
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    // Should return default scores
    expect(scores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });

  it('should handle missing fields in stored scores', () => {
    // Set partial data
    mockLocalStorage.setItem('rps-scores', JSON.stringify({ wins: 5 }));
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    // Should return default scores since structure is invalid
    expect(scores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });

  it('should handle wrong data types in stored scores', () => {
    // Set wrong types
    mockLocalStorage.setItem('rps-scores', JSON.stringify({ 
      wins: 'five', 
      losses: null, 
      ties: undefined 
    }));
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    // Should return default scores since types are invalid
    expect(scores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });


  it('should handle negative numbers in stored scores', () => {
    // Set negative numbers
    mockLocalStorage.setItem('rps-scores', JSON.stringify({ 
      wins: -5, 
      losses: -3, 
      ties: -2 
    }));
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    // Should accept negative numbers as they are valid numbers
    // (though semantically incorrect, the storage service validates type not value)
    expect(typeof scores.wins).toBe('number');
    expect(typeof scores.losses).toBe('number');
    expect(typeof scores.ties).toBe('number');
  });

  it('should handle empty localStorage', () => {
    // Ensure localStorage is empty
    mockLocalStorage.clear();
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    expect(scores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });

  it('should handle array instead of object in localStorage', () => {
    mockLocalStorage.setItem('rps-scores', JSON.stringify([1, 2, 3]));
    
    const storageService = new StorageService();
    const scores = storageService.getScores();
    
    // Should return default scores since it's not an object with correct structure
    expect(scores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });
});

/**
 * Edge case tests for rapid user interactions during animations
 * Validates: Requirements 2.4
 */
describe('Rapid User Interaction Edge Cases', () => {
  let mockLocalStorage: ReturnType<typeof createLocalStorageMock>;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    originalLocalStorage = global.localStorage;
    mockLocalStorage = createLocalStorageMock();
    (global as any).localStorage = mockLocalStorage;
  });

  afterEach(() => {
    (global as any).localStorage = originalLocalStorage;
    cleanup();
  });

  it('should ignore rapid clicks on the same button during animation', async () => {
    render(<GameBoard />);
    
    const rockButton = screen.getByTestId('choice-button-rock');
    
    // Rapid fire clicks
    fireEvent.click(rockButton);
    fireEvent.click(rockButton);
    fireEvent.click(rockButton);
    fireEvent.click(rockButton);
    fireEvent.click(rockButton);
    
    // Should only register one game
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });
    
    // Button should be disabled after first click
    expect(rockButton).toBeDisabled();
  });

  it('should ignore clicks on different buttons during animation', async () => {
    render(<GameBoard />);
    
    const rockButton = screen.getByTestId('choice-button-rock');
    const paperButton = screen.getByTestId('choice-button-paper');
    const scissorsButton = screen.getByTestId('choice-button-scissors');
    
    // Click rock first
    fireEvent.click(rockButton);
    
    // Try to click other buttons rapidly
    fireEvent.click(paperButton);
    fireEvent.click(scissorsButton);
    fireEvent.click(paperButton);
    
    // Should still be in selecting state for rock
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });
    
    // All buttons should be disabled
    expect(rockButton).toBeDisabled();
    expect(paperButton).toBeDisabled();
    expect(scissorsButton).toBeDisabled();
  });

  it('should handle clicks during result display phase', async () => {
    render(<GameBoard />);
    
    const rockButton = screen.getByTestId('choice-button-rock');
    fireEvent.click(rockButton);
    
    // Wait for result to show
    await waitFor(() => {
      const resultTexts = screen.queryAllByText(/You Win!|Computer Wins!|It's a Tie!/);
      expect(resultTexts.length).toBeGreaterThan(0);
    }, { timeout: 2000 });
    
    // Try clicking during result display
    fireEvent.click(rockButton);
    
    // Button should still be disabled
    expect(rockButton).toBeDisabled();
  });
});


/**
 * Edge case tests for component rendering with various prop combinations
 * Validates: Requirements 3.3, 2.4
 */
describe('Component Prop Combination Edge Cases', () => {
  afterEach(() => {
    cleanup();
  });

  it('should render ChoiceButton with all animation states', () => {
    const handleClick = () => {};
    const states: Array<'idle' | 'selecting' | 'revealing' | 'showing-result'> = [
      'idle', 'selecting', 'revealing', 'showing-result'
    ];
    
    states.forEach(state => {
      cleanup();
      render(
        <ChoiceButton
          choice="rock"
          onClick={handleClick}
          animationState={state}
        />
      );
      
      const button = screen.getByTestId('choice-button-rock');
      expect(button).toBeInTheDocument();
      
      if (state !== 'idle') {
        expect(button.className).toContain(`choice-button--${state}`);
      }
    });
  });

  it('should render ChoiceButton with disabled and selected combination', () => {
    const handleClick = () => {};
    
    render(
      <ChoiceButton
        choice="paper"
        onClick={handleClick}
        disabled={true}
        isSelected={true}
      />
    );
    
    const button = screen.getByTestId('choice-button-paper');
    expect(button).toBeDisabled();
    expect(button.className).toContain('choice-button--selected');
    expect(button.className).toContain('choice-button--disabled');
  });

  it('should not call onClick when ChoiceButton is disabled', () => {
    let clickCount = 0;
    const handleClick = () => { clickCount++; };
    
    render(
      <ChoiceButton
        choice="scissors"
        onClick={handleClick}
        disabled={true}
      />
    );
    
    const button = screen.getByTestId('choice-button-scissors');
    fireEvent.click(button);
    fireEvent.click(button);
    fireEvent.click(button);
    
    expect(clickCount).toBe(0);
  });

  it('should render all three choice types correctly', () => {
    const handleClick = () => {};
    const choices: Array<'rock' | 'paper' | 'scissors'> = ['rock', 'paper', 'scissors'];
    const expectedEmojis = { rock: '🪨', paper: '📄', scissors: '✂️' };
    
    choices.forEach(choice => {
      cleanup();
      render(
        <ChoiceButton
          choice={choice}
          onClick={handleClick}
        />
      );
      
      const button = screen.getByTestId(`choice-button-${choice}`);
      expect(button).toBeInTheDocument();
      expect(button.textContent).toContain(expectedEmojis[choice]);
      expect(button.textContent).toContain(choice.charAt(0).toUpperCase() + choice.slice(1));
    });
  });
});

describe('ScoreBoard Edge Cases', () => {
  let mockLocalStorage: ReturnType<typeof createLocalStorageMock>;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    originalLocalStorage = global.localStorage;
    mockLocalStorage = createLocalStorageMock();
    (global as any).localStorage = mockLocalStorage;
  });

  afterEach(() => {
    (global as any).localStorage = originalLocalStorage;
    cleanup();
  });

  it('should handle very large score numbers', async () => {
    const largeScores = { wins: 999999, losses: 888888, ties: 777777 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(largeScores));
    
    render(<ScoreBoard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('999999');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('888888');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('777777');
    });
  });

  it('should calculate win percentage correctly with edge values', async () => {
    // 100% win rate
    mockLocalStorage.setItem('rps-scores', JSON.stringify({ wins: 10, losses: 0, ties: 0 }));
    
    render(<ScoreBoard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('win-percentage')).toHaveTextContent('100%');
    });
  });

  it('should calculate 0% win rate correctly', async () => {
    mockLocalStorage.setItem('rps-scores', JSON.stringify({ wins: 0, losses: 10, ties: 5 }));
    
    render(<ScoreBoard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('win-percentage')).toHaveTextContent('0%');
    });
  });

  it('should handle reset cancellation correctly', async () => {
    const scores = { wins: 5, losses: 3, ties: 2 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(scores));
    
    render(<ScoreBoard />);
    
    // Wait for load
    await waitFor(() => {
      expect(screen.getByTestId('reset-button')).not.toBeDisabled();
    });
    
    // Click reset
    fireEvent.click(screen.getByTestId('reset-button'));
    
    // Verify confirmation dialog appears
    expect(screen.getByTestId('confirm-reset')).toBeInTheDocument();
    
    // Cancel
    fireEvent.click(screen.getByTestId('confirm-no'));
    
    // Verify dialog is gone and scores unchanged
    expect(screen.queryByTestId('confirm-reset')).not.toBeInTheDocument();
    expect(screen.getByTestId('wins-count')).toHaveTextContent('5');
  });
});
