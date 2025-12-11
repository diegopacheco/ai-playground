import React, { useState, useEffect } from 'react';
import { Choice, GameState } from '../types';
import { determineWinner, generateComputerChoice } from '../utils/gameLogic';
import { ChoiceButton } from './ChoiceButton';

interface GameBoardProps {
  onGameComplete?: (result: 'win' | 'lose' | 'tie') => void;
}

/**
 * Main game interface component that handles player choice selection,
 * computer opponent logic, and game flow management with state transitions
 */
export const GameBoard: React.FC<GameBoardProps> = ({ onGameComplete }) => {
  const [gameState, setGameState] = useState<GameState>({
    playerChoice: null,
    computerChoice: null,
    gameResult: null,
    isPlaying: false,
    animationState: 'idle'
  });

  // Handle player choice selection
  const handlePlayerChoice = (choice: Choice) => {
    if (gameState.isPlaying || gameState.animationState !== 'idle') {
      return; // Prevent multiple selections during game flow
    }

    // Set player choice and start game flow
    setGameState(prev => ({
      ...prev,
      playerChoice: choice,
      isPlaying: true,
      animationState: 'selecting'
    }));
  };

  // Game flow management with useEffect
  useEffect(() => {
    if (gameState.animationState === 'selecting' && gameState.playerChoice) {
      // After selection animation, generate computer choice and reveal
      const timer = setTimeout(() => {
        const computerChoice = generateComputerChoice();
        setGameState(prev => ({
          ...prev,
          computerChoice,
          animationState: 'revealing'
        }));
      }, 600); // Match CSS animation duration

      return () => clearTimeout(timer);
    }
  }, [gameState.animationState, gameState.playerChoice]);

  useEffect(() => {
    if (gameState.animationState === 'revealing' && gameState.computerChoice && gameState.playerChoice) {
      // After reveal animation, determine winner and show result
      const timer = setTimeout(() => {
        const result = determineWinner(gameState.playerChoice!, gameState.computerChoice!);
        setGameState(prev => ({
          ...prev,
          gameResult: result,
          animationState: 'showing-result'
        }));

        // Notify parent component of game completion
        if (onGameComplete) {
          onGameComplete(result);
        }
      }, 400); // Match CSS animation duration

      return () => clearTimeout(timer);
    }
  }, [gameState.animationState, gameState.computerChoice, gameState.playerChoice, onGameComplete]);

  useEffect(() => {
    if (gameState.animationState === 'showing-result') {
      // After showing result, reset for next round
      const timer = setTimeout(() => {
        setGameState(prev => ({
          ...prev,
          playerChoice: null,
          computerChoice: null,
          gameResult: null,
          isPlaying: false,
          animationState: 'idle'
        }));
      }, 2000); // Show result for 2 seconds

      return () => clearTimeout(timer);
    }
  }, [gameState.animationState]);

  // Get result message based on game outcome
  const getResultMessage = (): string => {
    if (!gameState.gameResult) return '';
    
    switch (gameState.gameResult) {
      case 'win':
        return 'You Win! 🎉';
      case 'lose':
        return 'Computer Wins! 🤖';
      case 'tie':
        return "It's a Tie! 🤝";
      default:
        return '';
    }
  };

  // Get choice emoji for display
  const getChoiceEmoji = (choice: Choice): string => {
    switch (choice) {
      case 'rock':
        return '🪨';
      case 'paper':
        return '📄';
      case 'scissors':
        return '✂️';
      default:
        return '';
    }
  };

  const choices: Choice[] = ['rock', 'paper', 'scissors'];

  return (
    <div className="gameboard">
      <div className="gameboard__header">
        <h2 className="gameboard__title">Choose Your Move</h2>
        {gameState.animationState !== 'idle' && (
          <div className="gameboard__status">
            {gameState.animationState === 'selecting' && 'Making your choice...'}
            {gameState.animationState === 'revealing' && 'Computer is choosing...'}
            {gameState.animationState === 'showing-result' && getResultMessage()}
          </div>
        )}
      </div>

      <div className="gameboard__choices">
        {choices.map((choice) => (
          <ChoiceButton
            key={choice}
            choice={choice}
            onClick={handlePlayerChoice}
            disabled={gameState.isPlaying}
            animationState={gameState.animationState}
            isSelected={gameState.playerChoice === choice}
          />
        ))}
      </div>

      {(gameState.playerChoice || gameState.computerChoice) && (
        <div className="gameboard__battle">
          <div className="gameboard__player-section">
            <h3 className="gameboard__section-title">Your Choice</h3>
            <div className="gameboard__choice-display">
              {gameState.playerChoice ? (
                <>
                  <span className="gameboard__choice-emoji">
                    {getChoiceEmoji(gameState.playerChoice)}
                  </span>
                  <span className="gameboard__choice-name">
                    {gameState.playerChoice.charAt(0).toUpperCase() + gameState.playerChoice.slice(1)}
                  </span>
                </>
              ) : (
                <span className="gameboard__choice-placeholder">?</span>
              )}
            </div>
          </div>

          <div className="gameboard__vs">VS</div>

          <div className="gameboard__computer-section">
            <h3 className="gameboard__section-title">Computer Choice</h3>
            <div className="gameboard__choice-display">
              {gameState.computerChoice ? (
                <>
                  <span className="gameboard__choice-emoji">
                    {getChoiceEmoji(gameState.computerChoice)}
                  </span>
                  <span className="gameboard__choice-name">
                    {gameState.computerChoice.charAt(0).toUpperCase() + gameState.computerChoice.slice(1)}
                  </span>
                </>
              ) : (
                <span className="gameboard__choice-placeholder">?</span>
              )}
            </div>
          </div>
        </div>
      )}

      {gameState.gameResult && gameState.animationState === 'showing-result' && (
        <div className="gameboard__result">
          <div className={`gameboard__result-message gameboard__result-message--${gameState.gameResult}`}>
            {getResultMessage()}
          </div>
        </div>
      )}
    </div>
  );
};