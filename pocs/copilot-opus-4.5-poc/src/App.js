import React, { useState, useEffect } from 'react';
import './App.css';

const CHOICES = ['rock', 'paper', 'scissors'];

const CHOICE_EMOJIS = {
  rock: '🪨',
  paper: '📄',
  scissors: '✂️'
};

const getWinner = (playerChoice, computerChoice) => {
  if (playerChoice === computerChoice) return 'draw';
  
  const winConditions = {
    rock: 'scissors',
    paper: 'rock',
    scissors: 'paper'
  };
  
  return winConditions[playerChoice] === computerChoice ? 'player' : 'computer';
};

const getResultMessage = (result) => {
  switch (result) {
    case 'player':
      return '🎉 You Win!';
    case 'computer':
      return '😢 You Lose!';
    case 'draw':
      return '🤝 It\'s a Draw!';
    default:
      return '';
  }
};

function App() {
  const [playerChoice, setPlayerChoice] = useState(null);
  const [computerChoice, setComputerChoice] = useState(null);
  const [result, setResult] = useState(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [stats, setStats] = useState(() => {
    const saved = localStorage.getItem('rpsStats');
    return saved ? JSON.parse(saved) : { wins: 0, losses: 0, draws: 0 };
  });
  const [gameHistory, setGameHistory] = useState(() => {
    const saved = localStorage.getItem('rpsHistory');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('rpsStats', JSON.stringify(stats));
  }, [stats]);

  useEffect(() => {
    localStorage.setItem('rpsHistory', JSON.stringify(gameHistory));
  }, [gameHistory]);

  const playGame = (choice) => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setPlayerChoice(choice);
    setResult(null);
    setComputerChoice(null);

    // Simulate computer "thinking"
    setTimeout(() => {
      const compChoice = CHOICES[Math.floor(Math.random() * CHOICES.length)];
      setComputerChoice(compChoice);
      
      const gameResult = getWinner(choice, compChoice);
      setResult(gameResult);
      
      // Update stats
      setStats(prev => ({
        ...prev,
        wins: gameResult === 'player' ? prev.wins + 1 : prev.wins,
        losses: gameResult === 'computer' ? prev.losses + 1 : prev.losses,
        draws: gameResult === 'draw' ? prev.draws + 1 : prev.draws
      }));
      
      // Update history (keep last 10 games)
      setGameHistory(prev => {
        const newHistory = [
          {
            playerChoice: choice,
            computerChoice: compChoice,
            result: gameResult,
            timestamp: new Date().toISOString()
          },
          ...prev
        ].slice(0, 10);
        return newHistory;
      });
      
      setIsAnimating(false);
    }, 1000);
  };

  const resetStats = () => {
    setStats({ wins: 0, losses: 0, draws: 0 });
    setGameHistory([]);
    setPlayerChoice(null);
    setComputerChoice(null);
    setResult(null);
    localStorage.removeItem('rpsStats');
    localStorage.removeItem('rpsHistory');
  };

  const totalGames = stats.wins + stats.losses + stats.draws;
  const winRate = totalGames > 0 ? ((stats.wins / totalGames) * 100).toFixed(1) : 0;

  return (
    <div className="app">
      <h1 className="title">🎮 Rock Paper Scissors</h1>
      
      <div className="stats-container">
        <div className="stat-box win">
          <span className="stat-label">Wins</span>
          <span className="stat-value">{stats.wins}</span>
        </div>
        <div className="stat-box loss">
          <span className="stat-label">Losses</span>
          <span className="stat-value">{stats.losses}</span>
        </div>
        <div className="stat-box draw">
          <span className="stat-label">Draws</span>
          <span className="stat-value">{stats.draws}</span>
        </div>
        <div className="stat-box rate">
          <span className="stat-label">Win Rate</span>
          <span className="stat-value">{winRate}%</span>
        </div>
      </div>

      <div className="game-area">
        <div className="choices-container">
          <div className="player-side">
            <h3>You</h3>
            <div className={`choice-display ${playerChoice ? 'active' : ''}`}>
              {playerChoice ? CHOICE_EMOJIS[playerChoice] : '❓'}
            </div>
          </div>
          
          <div className="vs">VS</div>
          
          <div className="computer-side">
            <h3>Computer</h3>
            <div className={`choice-display ${isAnimating ? 'thinking' : ''} ${computerChoice ? 'active' : ''}`}>
              {isAnimating ? '🤔' : (computerChoice ? CHOICE_EMOJIS[computerChoice] : '❓')}
            </div>
          </div>
        </div>

        {result && (
          <div className={`result-message ${result}`}>
            {getResultMessage(result)}
          </div>
        )}

        <div className="buttons-container">
          {CHOICES.map(choice => (
            <button
              key={choice}
              className={`choice-button ${playerChoice === choice ? 'selected' : ''}`}
              onClick={() => playGame(choice)}
              disabled={isAnimating}
            >
              <span className="button-emoji">{CHOICE_EMOJIS[choice]}</span>
              <span className="button-text">{choice.charAt(0).toUpperCase() + choice.slice(1)}</span>
            </button>
          ))}
        </div>
      </div>

      {gameHistory.length > 0 && (
        <div className="history-container">
          <h3>Recent Games</h3>
          <div className="history-list">
            {gameHistory.map((game, index) => (
              <div key={index} className={`history-item ${game.result}`}>
                <span>{CHOICE_EMOJIS[game.playerChoice]}</span>
                <span className="history-vs">vs</span>
                <span>{CHOICE_EMOJIS[game.computerChoice]}</span>
                <span className="history-result">
                  {game.result === 'player' ? '✅' : game.result === 'computer' ? '❌' : '➖'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <button className="reset-button" onClick={resetStats}>
        🔄 Reset All Stats
      </button>
    </div>
  );
}

export default App;
