import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'bun:test';
import App from './App';

describe('App', () => {
  it('renders the game title', () => {
    render(<App />);
    expect(screen.getByText('Rock Paper Scissors Game')).toBeInTheDocument();
  });

  it('renders the ScoreBoard component', () => {
    render(<App />);
    expect(screen.getByTestId('scoreboard')).toBeInTheDocument();
    expect(screen.getByText('Game Statistics')).toBeInTheDocument();
  });

  it('renders the GameBoard component with choice buttons', () => {
    render(<App />);
    expect(screen.getByText('Choose Your Move')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-rock')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-paper')).toBeInTheDocument();
    expect(screen.getByTestId('choice-button-scissors')).toBeInTheDocument();
  });
});