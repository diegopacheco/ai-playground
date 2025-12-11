import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ResultDisplay } from './ResultDisplay';

describe('ResultDisplay Component', () => {
  test('should not render when isVisible is false', () => {
    render(
      <ResultDisplay
        playerChoice="rock"
        computerChoice="scissors"
        gameResult="win"
        isVisible={false}
      />
    );
    
    expect(screen.queryByTestId('result-display')).not.toBeInTheDocument();
  });

  test('should not render when choices are null', () => {
    render(
      <ResultDisplay
        playerChoice={null}
        computerChoice={null}
        gameResult={null}
        isVisible={true}
      />
    );
    
    expect(screen.queryByTestId('result-display')).not.toBeInTheDocument();
  });

  test('should display win message when player wins', () => {
    render(
      <ResultDisplay
        playerChoice="rock"
        computerChoice="scissors"
        gameResult="win"
        isVisible={true}
      />
    );
    
    expect(screen.getByTestId('result-message')).toHaveTextContent('You Win!');
  });

  test('should display lose message when computer wins', () => {
    render(
      <ResultDisplay
        playerChoice="rock"
        computerChoice="paper"
        gameResult="lose"
        isVisible={true}
      />
    );
    
    expect(screen.getByTestId('result-message')).toHaveTextContent('Computer Wins!');
  });

  test('should display tie message on tie', () => {
    render(
      <ResultDisplay
        playerChoice="rock"
        computerChoice="rock"
        gameResult="tie"
        isVisible={true}
      />
    );
    
    expect(screen.getByTestId('result-message')).toHaveTextContent("It's a Tie!");
  });


  test('should display both player and computer choices with emojis', () => {
    render(
      <ResultDisplay
        playerChoice="paper"
        computerChoice="rock"
        gameResult="win"
        isVisible={true}
      />
    );
    
    const playerDisplay = screen.getByTestId('player-choice-display');
    const computerDisplay = screen.getByTestId('computer-choice-display');
    
    expect(playerDisplay).toHaveTextContent('📄');
    expect(playerDisplay).toHaveTextContent('Paper');
    expect(computerDisplay).toHaveTextContent('🪨');
    expect(computerDisplay).toHaveTextContent('Rock');
  });

  test('should highlight winning choice when player wins', () => {
    render(
      <ResultDisplay
        playerChoice="scissors"
        computerChoice="paper"
        gameResult="win"
        isVisible={true}
      />
    );
    
    const playerDisplay = screen.getByTestId('player-choice-display');
    const computerDisplay = screen.getByTestId('computer-choice-display');
    
    expect(playerDisplay).toHaveClass('result-display__choice--winner');
    expect(computerDisplay).toHaveClass('result-display__choice--loser');
  });

  test('should highlight winning choice when computer wins', () => {
    render(
      <ResultDisplay
        playerChoice="paper"
        computerChoice="scissors"
        gameResult="lose"
        isVisible={true}
      />
    );
    
    const playerDisplay = screen.getByTestId('player-choice-display');
    const computerDisplay = screen.getByTestId('computer-choice-display');
    
    expect(playerDisplay).toHaveClass('result-display__choice--loser');
    expect(computerDisplay).toHaveClass('result-display__choice--winner');
  });

  test('should not highlight any choice on tie', () => {
    render(
      <ResultDisplay
        playerChoice="rock"
        computerChoice="rock"
        gameResult="tie"
        isVisible={true}
      />
    );
    
    const playerDisplay = screen.getByTestId('player-choice-display');
    const computerDisplay = screen.getByTestId('computer-choice-display');
    
    expect(playerDisplay).not.toHaveClass('result-display__choice--winner');
    expect(playerDisplay).not.toHaveClass('result-display__choice--loser');
    expect(computerDisplay).not.toHaveClass('result-display__choice--winner');
    expect(computerDisplay).not.toHaveClass('result-display__choice--loser');
  });
});
