import { describe, it, expect, afterEach } from 'bun:test';
import { render, cleanup } from '@testing-library/react';
import React from 'react';
import { ChoiceButton } from './ChoiceButton';
import '../test-setup';

describe('ChoiceButton Integration Tests', () => {
  afterEach(() => {
    cleanup();
  });

  it('should render with different animation states', () => {
    const handleClick = () => {};
    
    // Test idle state
    const { rerender } = render(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
        animationState="idle"
      />
    );

    let button = document.querySelector('[data-testid="choice-button-rock"]');
    expect(button?.className).toContain('choice-button');

    // Test selecting state
    rerender(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
        animationState="selecting"
      />
    );

    button = document.querySelector('[data-testid="choice-button-rock"]');
    expect(button?.className).toContain('choice-button--selecting');

    // Test revealing state
    rerender(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
        animationState="revealing"
      />
    );

    button = document.querySelector('[data-testid="choice-button-rock"]');
    expect(button?.className).toContain('choice-button--revealing');

    // Test showing-result state
    rerender(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
        animationState="showing-result"
      />
    );

    button = document.querySelector('[data-testid="choice-button-rock"]');
    expect(button?.className).toContain('choice-button--showing-result');
  });

  it('should render with selected state', () => {
    const handleClick = () => {};
    
    render(
      <ChoiceButton
        choice="paper"
        onClick={handleClick}
        isSelected={true}
      />
    );

    const button = document.querySelector('[data-testid="choice-button-paper"]');
    expect(button?.className).toContain('choice-button--selected');
  });

  it('should render all choice types correctly', () => {
    const handleClick = () => {};
    
    // Test rock
    const { rerender } = render(
      <ChoiceButton
        choice="rock"
        onClick={handleClick}
      />
    );

    let button = document.querySelector('[data-testid="choice-button-rock"]');
    expect(button?.textContent).toContain('Rock');
    expect(button?.textContent).toContain('🪨');

    // Test paper
    rerender(
      <ChoiceButton
        choice="paper"
        onClick={handleClick}
      />
    );

    button = document.querySelector('[data-testid="choice-button-paper"]');
    expect(button?.textContent).toContain('Paper');
    expect(button?.textContent).toContain('📄');

    // Test scissors
    rerender(
      <ChoiceButton
        choice="scissors"
        onClick={handleClick}
      />
    );

    button = document.querySelector('[data-testid="choice-button-scissors"]');
    expect(button?.textContent).toContain('Scissors');
    expect(button?.textContent).toContain('✂️');
  });
});