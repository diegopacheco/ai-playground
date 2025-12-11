import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, test, expect, mock } from 'bun:test';
import '@testing-library/jest-dom';
import { GameBoard } from './GameBoard';

describe('GameBoard Component', () => {
  test('should render choice buttons and title', () => {
    render(<GameBoard />);
    
    expect(screen.getByText('Choose Your Move')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-rock')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-paper')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-scissors')).toBeInTheDocument();
  });

  test('should handle player choice selection and game flow', async () => {
    const mockOnGameComplete = mock(() => {});
    render(<GameBoard onGameComplete={mockOnGameComplete} />);
    
    // Click rock button
    const rockButton = screen.getByTestId('choice-button-rock');
    fireEvent.click(rockButton);
    
    // Should show selecting status
    expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    
    // Wait for game flow to complete
    await waitFor(() => {
      expect(mockOnGameComplete).toHaveBeenCalledWith(
        expect.stringMatching(/^(win|lose|tie)$/)
      );
    }, { timeout: 3000 });
  });

  test('should display player and computer choices during battle', async () => {
    render(<GameBoard />);
    
    // Click paper button
    const paperButton = screen.getByTestId('choice-button-paper');
    fireEvent.click(paperButton);
    
    // Should show player choice section
    await waitFor(() => {
      expect(screen.getByText('Your Choice')).toBeInTheDocument();
      expect(screen.getByText('Computer Choice')).toBeInTheDocument();
    });
  });

  test('should prevent multiple selections during game flow', () => {
    render(<GameBoard />);
    
    const rockButton = screen.getByTestId('choice-button-rock');
    const paperButton = screen.getByTestId('choice-button-paper');
    
    // Click rock button
    fireEvent.click(rockButton);
    
    // Try to click paper button (should be disabled)
    fireEvent.click(paperButton);
    
    // Should still show selecting status for rock
    expect(screen.getByText('Making your choice...')).toBeInTheDocument();
  });

  test('should reset to idle state after showing result', async () => {
    render(<GameBoard />);
    
    // Click scissors button
    const scissorsButton = screen.getByTestId('choice-button-scissors');
    fireEvent.click(scissorsButton);
    
    // Wait for selecting state
    await waitFor(() => {
      expect(screen.getByText('Making your choice...')).toBeInTheDocument();
    });
    
    // Wait for computer choosing state
    await waitFor(() => {
      expect(screen.getByText('Computer is choosing...')).toBeInTheDocument();
    }, { timeout: 1000 });
    
    // Wait for result to appear and then disappear (reset to idle)
    // Total animation time: 600ms (selecting) + 400ms (revealing) + 2000ms (showing-result) = 3000ms
    // Adding extra buffer for test stability
    await waitFor(() => {
      expect(screen.getByText('Choose Your Move')).toBeInTheDocument();
      expect(scissorsButton).not.toBeDisabled();
    }, { timeout: 5000 });
  });
});
