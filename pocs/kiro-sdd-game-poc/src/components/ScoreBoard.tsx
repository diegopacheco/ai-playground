import React, { useState, useEffect } from 'react';
import { ScoreState } from '../types';
import { StorageService } from '../services/StorageService';

interface ScoreBoardProps {
  onScoreUpdate?: (scores: ScoreState) => void;
}

/**
 * ScoreBoard component displays current game statistics and provides reset functionality
 * Integrates with localStorage service for persistent score tracking
 */
export const ScoreBoard: React.FC<ScoreBoardProps> = ({ onScoreUpdate }) => {
  const [scores, setScores] = useState<ScoreState>({ wins: 0, losses: 0, ties: 0 });
  const [showConfirmReset, setShowConfirmReset] = useState(false);
  const [storageService] = useState(() => new StorageService());

  // Load scores from storage on component mount
  useEffect(() => {
    const loadedScores = storageService.getScores();
    setScores(loadedScores);
    onScoreUpdate?.(loadedScores);
  }, [storageService, onScoreUpdate]);

  // Handle score reset with confirmation
  const handleResetRequest = () => {
    setShowConfirmReset(true);
  };

  const handleConfirmReset = () => {
    const resetScores = { wins: 0, losses: 0, ties: 0 };
    storageService.resetScores();
    setScores(resetScores);
    setShowConfirmReset(false);
    onScoreUpdate?.(resetScores);
  };

  const handleCancelReset = () => {
    setShowConfirmReset(false);
  };

  // Update scores (to be called by parent component)
  const updateScores = (newScores: ScoreState) => {
    storageService.saveScores(newScores);
    setScores(newScores);
    onScoreUpdate?.(newScores);
  };

  // Note: This component doesn't use ref forwarding
  // Use ScoreBoardWithRef for ref-based score updates

  const totalGames = scores.wins + scores.losses + scores.ties;
  const winPercentage = totalGames > 0 ? Math.round((scores.wins / totalGames) * 100) : 0;

  return (
    <div className="scoreboard" data-testid="scoreboard">
      <h2 className="scoreboard__title">Game Statistics</h2>
      
      <div className="scoreboard__stats">
        <div className="scoreboard__stat scoreboard__stat--wins">
          <span className="scoreboard__stat-label">Wins</span>
          <span className="scoreboard__stat-value" data-testid="wins-count">
            {scores.wins}
          </span>
        </div>
        
        <div className="scoreboard__stat scoreboard__stat--losses">
          <span className="scoreboard__stat-label">Losses</span>
          <span className="scoreboard__stat-value" data-testid="losses-count">
            {scores.losses}
          </span>
        </div>
        
        <div className="scoreboard__stat scoreboard__stat--ties">
          <span className="scoreboard__stat-label">Ties</span>
          <span className="scoreboard__stat-value" data-testid="ties-count">
            {scores.ties}
          </span>
        </div>
      </div>

      {totalGames > 0 && (
        <div className="scoreboard__summary">
          <p className="scoreboard__total">
            Total Games: <span data-testid="total-games">{totalGames}</span>
          </p>
          <p className="scoreboard__percentage">
            Win Rate: <span data-testid="win-percentage">{winPercentage}%</span>
          </p>
        </div>
      )}

      <div className="scoreboard__actions">
        {!showConfirmReset ? (
          <button
            className="scoreboard__reset-button"
            onClick={handleResetRequest}
            data-testid="reset-button"
            disabled={totalGames === 0}
          >
            Reset Statistics
          </button>
        ) : (
          <div className="scoreboard__confirm-reset" data-testid="confirm-reset">
            <p className="scoreboard__confirm-message">
              Are you sure you want to reset all statistics?
            </p>
            <div className="scoreboard__confirm-actions">
              <button
                className="scoreboard__confirm-button scoreboard__confirm-button--yes"
                onClick={handleConfirmReset}
                data-testid="confirm-yes"
              >
                Yes, Reset
              </button>
              <button
                className="scoreboard__confirm-button scoreboard__confirm-button--no"
                onClick={handleCancelReset}
                data-testid="confirm-no"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Export a version with ref forwarding for parent component access
export const ScoreBoardWithRef = React.forwardRef<
  { updateScores: (scores: ScoreState) => void },
  ScoreBoardProps
>((props, ref) => {
  const [scores, setScores] = useState<ScoreState>({ wins: 0, losses: 0, ties: 0 });
  const [showConfirmReset, setShowConfirmReset] = useState(false);
  const [storageService] = useState(() => new StorageService());

  // Load scores from storage on component mount only
  useEffect(() => {
    const loadedScores = storageService.getScores();
    setScores(loadedScores);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageService]);

  // Handle score reset with confirmation
  const handleResetRequest = () => {
    setShowConfirmReset(true);
  };

  const handleConfirmReset = () => {
    const resetScores = { wins: 0, losses: 0, ties: 0 };
    storageService.resetScores();
    setScores(resetScores);
    setShowConfirmReset(false);
    props.onScoreUpdate?.(resetScores);
  };

  const handleCancelReset = () => {
    setShowConfirmReset(false);
  };

  // Update scores (to be called by parent component)
  const updateScores = (newScores: ScoreState) => {
    storageService.saveScores(newScores);
    setScores(newScores);
    props.onScoreUpdate?.(newScores);
  };

  // Expose updateScores method via ref
  React.useImperativeHandle(ref, () => ({
    updateScores
  }));

  const totalGames = scores.wins + scores.losses + scores.ties;
  const winPercentage = totalGames > 0 ? Math.round((scores.wins / totalGames) * 100) : 0;

  return (
    <div className="scoreboard" data-testid="scoreboard">
      <h2 className="scoreboard__title">Game Statistics</h2>
      
      <div className="scoreboard__stats">
        <div className="scoreboard__stat scoreboard__stat--wins">
          <span className="scoreboard__stat-label">Wins</span>
          <span className="scoreboard__stat-value" data-testid="wins-count">
            {scores.wins}
          </span>
        </div>
        
        <div className="scoreboard__stat scoreboard__stat--losses">
          <span className="scoreboard__stat-label">Losses</span>
          <span className="scoreboard__stat-value" data-testid="losses-count">
            {scores.losses}
          </span>
        </div>
        
        <div className="scoreboard__stat scoreboard__stat--ties">
          <span className="scoreboard__stat-label">Ties</span>
          <span className="scoreboard__stat-value" data-testid="ties-count">
            {scores.ties}
          </span>
        </div>
      </div>

      {totalGames > 0 && (
        <div className="scoreboard__summary">
          <p className="scoreboard__total">
            Total Games: <span data-testid="total-games">{totalGames}</span>
          </p>
          <p className="scoreboard__percentage">
            Win Rate: <span data-testid="win-percentage">{winPercentage}%</span>
          </p>
        </div>
      )}

      <div className="scoreboard__actions">
        {!showConfirmReset ? (
          <button
            className="scoreboard__reset-button"
            onClick={handleResetRequest}
            data-testid="reset-button"
            disabled={totalGames === 0}
          >
            Reset Statistics
          </button>
        ) : (
          <div className="scoreboard__confirm-reset" data-testid="confirm-reset">
            <p className="scoreboard__confirm-message">
              Are you sure you want to reset all statistics?
            </p>
            <div className="scoreboard__confirm-actions">
              <button
                className="scoreboard__confirm-button scoreboard__confirm-button--yes"
                onClick={handleConfirmReset}
                data-testid="confirm-yes"
              >
                Yes, Reset
              </button>
              <button
                className="scoreboard__confirm-button scoreboard__confirm-button--no"
                onClick={handleCancelReset}
                data-testid="confirm-no"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

ScoreBoardWithRef.displayName = 'ScoreBoardWithRef';