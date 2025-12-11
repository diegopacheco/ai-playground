import React, { useState, useRef, Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { GameBoard, ScoreBoardWithRef } from './components';
import type { ScoreState } from './types';

/**
 * Error Boundary component for graceful error handling
 * Catches JavaScript errors in child component tree and displays fallback UI
 */
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Game error caught by boundary:', error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  override render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>An error occurred while playing the game.</p>
          <button onClick={this.handleRetry} className="error-boundary__retry">
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Main App component that coordinates all child components
 * Implements overall game state management using React hooks
 * Handles component communication and state updates
 */
function App() {
  const [scores, setScores] = useState<ScoreState>({ wins: 0, losses: 0, ties: 0 });
  const scoreBoardRef = useRef<{ updateScores: (scores: ScoreState) => void }>(null);

  /**
   * Handle game completion by updating scores through ScoreBoard ref
   * This enables communication between GameBoard and ScoreBoard components
   */
  const handleGameComplete = (result: 'win' | 'lose' | 'tie') => {
    setScores(prevScores => {
      const newScores = { ...prevScores };
      switch (result) {
        case 'win':
          newScores.wins += 1;
          break;
        case 'lose':
          newScores.losses += 1;
          break;
        case 'tie':
          newScores.ties += 1;
          break;
      }
      // Update ScoreBoard via ref
      if (scoreBoardRef.current) {
        scoreBoardRef.current.updateScores(newScores);
      }
      return newScores;
    });
  };

  return (
    <ErrorBoundary>
      <div className="app">
        <h1>Rock Paper Scissors Game</h1>
        <ErrorBoundary fallback={<div className="error-fallback">Score tracking unavailable</div>}>
          <ScoreBoardWithRef ref={scoreBoardRef} />
        </ErrorBoundary>
        <ErrorBoundary fallback={<div className="error-fallback">Game board unavailable</div>}>
          <GameBoard onGameComplete={handleGameComplete} />
        </ErrorBoundary>
      </div>
    </ErrorBoundary>
  );
}

export default App;