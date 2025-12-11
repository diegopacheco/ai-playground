import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, mock } from 'bun:test';
import { ScoreBoard, ScoreBoardWithRef } from './ScoreBoard';
import { ScoreState } from '../types';

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

describe('ScoreBoard Component', () => {
  let mockLocalStorage: ReturnType<typeof createLocalStorageMock>;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    // Save original and set up mock
    originalLocalStorage = global.localStorage;
    mockLocalStorage = createLocalStorageMock();
    (global as any).localStorage = mockLocalStorage;
  });

  afterEach(() => {
    // Restore original
    (global as any).localStorage = originalLocalStorage;
  });

  it('renders with initial zero scores', () => {
    render(<ScoreBoard />);
    
    expect(screen.getByTestId('wins-count')).toHaveTextContent('0');
    expect(screen.getByTestId('losses-count')).toHaveTextContent('0');
    expect(screen.getByTestId('ties-count')).toHaveTextContent('0');
  });

  it('displays existing scores from localStorage', async () => {
    const existingScores = { wins: 5, losses: 3, ties: 2 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));

    render(<ScoreBoard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('5');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('3');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('2');
    });
  });

  it('shows summary statistics when games have been played', async () => {
    const existingScores = { wins: 6, losses: 3, ties: 1 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard />);
    
    await waitFor(() => {
      expect(screen.getByTestId('total-games')).toHaveTextContent('10');
      expect(screen.getByTestId('win-percentage')).toHaveTextContent('60%');
    });
  });

  it('does not show summary when no games have been played', () => {
    render(<ScoreBoard />);
    
    expect(screen.queryByTestId('total-games')).not.toBeInTheDocument();
    expect(screen.queryByTestId('win-percentage')).not.toBeInTheDocument();
  });

  it('disables reset button when no games have been played', () => {
    render(<ScoreBoard />);
    
    const resetButton = screen.getByTestId('reset-button');
    expect(resetButton).toBeDisabled();
  });

  it('enables reset button when games have been played', async () => {
    const existingScores = { wins: 1, losses: 0, ties: 0 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard />);
    
    await waitFor(() => {
      const resetButton = screen.getByTestId('reset-button');
      expect(resetButton).not.toBeDisabled();
    });
  });

  it('shows confirmation dialog when reset is requested', async () => {
    const existingScores = { wins: 1, losses: 0, ties: 0 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard />);
    
    // Wait for scores to load and button to be enabled
    await waitFor(() => {
      expect(screen.getByTestId('reset-button')).not.toBeDisabled();
    });
    
    const resetButton = screen.getByTestId('reset-button');
    fireEvent.click(resetButton);
    
    expect(screen.getByTestId('confirm-reset')).toBeInTheDocument();
    expect(screen.getByText('Are you sure you want to reset all statistics?')).toBeInTheDocument();
  });


  it('resets scores when confirmed', async () => {
    const existingScores = { wins: 5, losses: 3, ties: 2 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard />);
    
    // Wait for scores to load and button to be enabled
    await waitFor(() => {
      expect(screen.getByTestId('reset-button')).not.toBeDisabled();
    });
    
    // Click reset button
    const resetButton = screen.getByTestId('reset-button');
    fireEvent.click(resetButton);
    
    // Confirm reset
    const confirmButton = screen.getByTestId('confirm-yes');
    fireEvent.click(confirmButton);
    
    // Check that scores are reset
    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('0');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('0');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('0');
    });
    
    // Check that localStorage is updated
    const storedScores = JSON.parse(mockLocalStorage.getItem('rps-scores') || '{}');
    expect(storedScores).toEqual({ wins: 0, losses: 0, ties: 0 });
  });

  it('cancels reset when cancelled', async () => {
    const existingScores = { wins: 5, losses: 3, ties: 2 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard />);
    
    // Wait for scores to load and button to be enabled
    await waitFor(() => {
      expect(screen.getByTestId('reset-button')).not.toBeDisabled();
    });
    
    // Click reset button
    const resetButton = screen.getByTestId('reset-button');
    fireEvent.click(resetButton);
    
    // Cancel reset
    const cancelButton = screen.getByTestId('confirm-no');
    fireEvent.click(cancelButton);
    
    // Check that confirmation dialog is hidden
    expect(screen.queryByTestId('confirm-reset')).not.toBeInTheDocument();
    
    // Check that scores are unchanged
    expect(screen.getByTestId('wins-count')).toHaveTextContent('5');
    expect(screen.getByTestId('losses-count')).toHaveTextContent('3');
    expect(screen.getByTestId('ties-count')).toHaveTextContent('2');
  });

  it('calls onScoreUpdate callback when provided', async () => {
    const mockCallback = mock(() => {});
    const existingScores = { wins: 2, losses: 1, ties: 1 };
    mockLocalStorage.setItem('rps-scores', JSON.stringify(existingScores));
    
    render(<ScoreBoard onScoreUpdate={mockCallback} />);
    
    await waitFor(() => {
      expect(mockCallback).toHaveBeenCalledWith(existingScores);
    });
  });
});

describe('ScoreBoardWithRef Component', () => {
  let mockLocalStorage: ReturnType<typeof createLocalStorageMock>;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    originalLocalStorage = global.localStorage;
    mockLocalStorage = createLocalStorageMock();
    (global as any).localStorage = mockLocalStorage;
  });

  afterEach(() => {
    (global as any).localStorage = originalLocalStorage;
  });

  it('allows updating scores via ref', async () => {
    const ref = React.createRef<{ updateScores: (scores: ScoreState) => void }>();
    
    render(<ScoreBoardWithRef ref={ref} />);
    
    // Update scores via ref
    const newScores = { wins: 3, losses: 1, ties: 0 };
    ref.current?.updateScores(newScores);
    
    // Check that scores are updated
    await waitFor(() => {
      expect(screen.getByTestId('wins-count')).toHaveTextContent('3');
      expect(screen.getByTestId('losses-count')).toHaveTextContent('1');
      expect(screen.getByTestId('ties-count')).toHaveTextContent('0');
    });
    
    // Check that localStorage is updated
    const storedScores = JSON.parse(mockLocalStorage.getItem('rps-scores') || '{}');
    expect(storedScores).toEqual(newScores);
  });
});
