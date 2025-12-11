import React from 'react';
import { Choice } from '../types';

interface ResultDisplayProps {
  playerChoice: Choice | null;
  computerChoice: Choice | null;
  gameResult: 'win' | 'lose' | 'tie' | null;
  isVisible: boolean;
}

/**
 * Component to show round results and winner announcements
 * Displays both player and computer choices with visual representations
 * Shows result messages for win/lose/tie scenarios
 */
export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  playerChoice,
  computerChoice,
  gameResult,
  isVisible
}) => {
  if (!isVisible || !playerChoice || !computerChoice || !gameResult) {
    return null;
  }

  // Get emoji representation for the choice
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

  // Get display name for the choice
  const getChoiceDisplayName = (choice: Choice): string => {
    return choice.charAt(0).toUpperCase() + choice.slice(1);
  };

  // Get result message based on game outcome
  const getResultMessage = (): string => {
    switch (gameResult) {
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


  // Determine if a choice is the winner for highlighting
  const isWinningChoice = (choice: Choice, isPlayer: boolean): boolean => {
    if (gameResult === 'tie') return false;
    if (gameResult === 'win' && isPlayer) return true;
    if (gameResult === 'lose' && !isPlayer) return true;
    return false;
  };

  // Get CSS class for choice display based on win/lose state
  const getChoiceClass = (choice: Choice, isPlayer: boolean): string => {
    const baseClass = 'result-display__choice';
    const classes = [baseClass];
    
    if (isWinningChoice(choice, isPlayer)) {
      classes.push('result-display__choice--winner');
    } else if (gameResult !== 'tie') {
      classes.push('result-display__choice--loser');
    }
    
    return classes.join(' ');
  };

  return (
    <div className="result-display" data-testid="result-display">
      <div className="result-display__choices">
        <div className={getChoiceClass(playerChoice, true)} data-testid="player-choice-display">
          <span className="result-display__label">You</span>
          <span className="result-display__emoji" aria-hidden="true">
            {getChoiceEmoji(playerChoice)}
          </span>
          <span className="result-display__choice-name">
            {getChoiceDisplayName(playerChoice)}
          </span>
        </div>

        <div className="result-display__vs">VS</div>

        <div className={getChoiceClass(computerChoice, false)} data-testid="computer-choice-display">
          <span className="result-display__label">Computer</span>
          <span className="result-display__emoji" aria-hidden="true">
            {getChoiceEmoji(computerChoice)}
          </span>
          <span className="result-display__choice-name">
            {getChoiceDisplayName(computerChoice)}
          </span>
        </div>
      </div>

      <div 
        className={`result-display__message result-display__message--${gameResult}`}
        data-testid="result-message"
      >
        {getResultMessage()}
      </div>
    </div>
  );
};
