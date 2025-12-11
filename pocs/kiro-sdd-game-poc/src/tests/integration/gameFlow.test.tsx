import React from 'react';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import '@testing-library/jest-dom';
import App from '../../App';
import { GameBoard } from '../../components/GameBoard';
import { ScoreBoard } from '../../components/ScoreBoard';

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
 * Integration tests for complete game flows
 * Tests full user journey from game start to score persistence
 * Validates: Requirements 1.1, 1.2, 1.3, 2.1, 3.1
 */
describe('Game Flow Integration Tests', () => {
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

  it('should complete a full game round from choice to result display', async () => {
    render(<GameBoard />);

    // Initial state - buttons should be enabled
    const rockButton = screen.getByTestId('choice-button-rock');
    expect(rockButton).not.toBeDisabled();

    // Make a choice
    fireEvent.click(rockButton);

    // Should transition to selecting state
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });

    // Should show player choice
    await waitFor(() => {
      expect(screen.getByText('Your Choice')).toBeInTheDocument();
    });

    // Should transition to revealing state
    await waitFor(() => {
      expect(screen.getByText('Computer is choosing...')).toBeInTheDocument();
    }, { timeout: 1000 });

    // Should show computer choice
    await waitFor(() => {
      expect(screen.getByText('Computer Choice')).toBeInTheDocument();
    });

    // Should show result (win, lose, or tie) - use getAllByText since result appears in multiple places
    await waitFor(() => {
      const resultTexts = screen.queryAllByText(/You Win!|Computer Wins!|It's a Tie!/);
      expect(resultTexts.length).toBeGreaterThan(0);
    }, { timeout: 2000 });
  });

  it('should disable all choice buttons during game animation', async () => {
    render(<GameBoard />);

    const rockButton = screen.getByTestId('choice-button-rock');
    const paperButton = screen.getByTestId('choice-button-paper');
    const scissorsButton = screen.getByTestId('choice-button-scissors');

    // Initially all buttons should be enabled
    expect(rockButton).not.toBeDisabled();
    expect(paperButton).not.toBeDisabled();
    expect(scissorsButton).not.toBeDisabled();

    // Make a choice
    fireEvent.click(rockButton);

    // All buttons should be disabled during animation
    await waitFor(() => {
      expect(rockButton).toBeDisabled();
      expect(paperButton).toBeDisabled();
      expect(scissorsButton).toBeDisabled();
    });
  });

  it('should reset to idle state and re-enable buttons after game completes', async () => {
    render(<GameBoard />);

    const rockButton = screen.getByTestId('choice-button-rock');
    fireEvent.click(rockButton);

    // Wait for game to complete and reset
    await waitFor(() => {
      expect(rockButton).not.toBeDisabled();
    }, { timeout: 5000 });

    // Should be able to start a new game
    fireEvent.click(rockButton);
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });
  });
});

describe('Score Persistence Integration Tests', () => {
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

  it('should load existing scores from localStorage on mount', async () => {
    const existingScores = { wins: 10, losses: 5, ties: 3 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));

    render(<ScoreBoard />);

    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('10');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('5');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('3');
    });
  });


  it('should persist score reset to localStorage', async () => {
    const existingScores = { wins: 5, losses: 3, ties: 2 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));

    render(<ScoreBoard />);

    // Wait for scores to load
    await waitFor(() => {
      expect(screen.getByTestId('reset-button')).not.toBeDisabled();
    });

    // Click reset
    fireEvent.click(screen.getByTestId('reset-button'));
    
    // Confirm reset
    fireEvent.click(screen.getByTestId('confirm-yes'));

    // Verify UI is updated
    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('0');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('0');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('0');
    });

    // Verify localStorage is updated
    const storedScores = JSON.parse(mockLocalStorage.getItem('rps-scores') || '{}');
    expect(storedScores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });

  it('should initialize with zero scores when localStorage is empty', () => {
    render(<ScoreBoard />);

    expect(screen.getByTestId('wins-count')).toHaveTextContent('0');
    expect(screen.getByTestId('losses-count')).toHaveTextContent('0');
    expect(screen.getByTestId('ties-count')).toHaveTextContent('0');
  });
});

describe('Animation Sequence Integration Tests', () => {
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

  it('should progress through all animation states in correct order', async () => {
    render(<GameBoard />);

    const scissorsButton = screen.getByTestId('choice-button-scissors');
    fireEvent.click(scissorsButton);

    // State 1: selecting
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });

    // State 2: revealing
    await waitFor(() => {
      expect(screen.getByText('Computer is choosing...')).toBeInTheDocument();
    }, { timeout: 1000 });

    // State 3: showing-result - use getAllByText since result appears in multiple places
    await waitFor(() => {
      const resultTexts = screen.queryAllByText(/You Win!|Computer Wins!|It's a Tie!/);
      expect(resultTexts.length).toBeGreaterThan(0);
    }, { timeout: 1500 });

    // State 4: back to idle
    await waitFor(() => {
      expect(scissorsButton).not.toBeDisabled();
    }, { timeout: 5000 });
  });

  it('should show VS indicator during battle phase', async () => {
    render(<GameBoard />);

    const paperButton = screen.getByTestId('choice-button-paper');
    fireEvent.click(paperButton);

    // VS should appear during battle
    await waitFor(() => {
      expect(screen.getByText('VS')).toBeInTheDocument();
    });
  });
});
