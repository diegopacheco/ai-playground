import { describe, it, expect, afterEach } from 'bun:test';
import { render, fireEvent, screen, cleanup } from '@testing-library/react';
import React from 'react';
import { ChoiceButton } from './ChoiceButton';
import '../test-setup';

describe('ChoiceButton Unit Tests', () => {
  afterEach(() => {
    cleanup();
  });
  it('should render correctly', () => {
    const handleClick = () => {};
    
    render(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
      />
    );

    const button = screen.getByTestId('choice-button-rock');
    expect(button).toBeTruthy();
  });

  it('should call onClick with correct choice when clicked', () => {
    let capturedChoice = null;
    const handleClick = (choice) => {
      capturedChoice = choice;
    };
    
    render(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
      />
    );

    const button = screen.getByTestId('choice-button-rock');
    fireEvent.click(button);
    
    expect(capturedChoice).toBe('rock');
  });
});