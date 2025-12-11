import React from 'react';
import { Choice } from '../types';

interface ChoiceButtonProps {
  choice: Choice;
  onClick: (choice: Choice) => void;
  disabled?: boolean;
  animationState?: 'idle' | 'selecting' | 'revealing' | 'showing-result';
  isSelected?: boolean;
}

/**
 * Reusable button component for rock/paper/scissors selections
 * Handles click events, visual feedback, and animation states
 */
export const ChoiceButton: React.FC<ChoiceButtonProps> = ({
  choice,
  onClick,
  disabled = false,
  animationState = 'idle',
  isSelected = false
}) => {
  const handleClick = () => {
    if (!disabled) {
      onClick(choice);
    }
  };

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

  // Determine CSS classes based on state
  const getButtonClasses = (): string => {
    const baseClasses = 'choice-button';
    const classes = [baseClasses];

    if (disabled) {
      classes.push('choice-button--disabled');
    }

    if (isSelected) {
      classes.push('choice-button--selected');
    }

    if (animationState !== 'idle') {
      classes.push(`choice-button--${animationState}`);
    }

    return classes.join(' ');
  };

  return (
    <button
      className={getButtonClasses()}
      onClick={handleClick}
      disabled={disabled}
      data-testid={`choice-button-${choice}`}
      aria-label={`Select ${getChoiceDisplayName(choice)}`}
    >
      <span className="choice-button__emoji" aria-hidden="true">
        {getChoiceEmoji(choice)}
      </span>
      <span className="choice-button__text">
        {getChoiceDisplayName(choice)}
      </span>
    </button>
  );
};