import { useState, useEffect } from 'react'

const CHOICES = ['rock', 'paper', 'scissors'];

function App() {
  const [userChoice, setUserChoice] = useState(null);
  const [computerChoice, setComputerChoice] = useState(null);
  const [result, setResult] = useState(null);
  const [score, setScore] = useState({
    wins: 0,
    losses: 0,
    draws: 0
  });

  // Load score from local storage on mount
  useEffect(() => {
    const savedScore = localStorage.getItem('rpsScore');
    if (savedScore) {
      setScore(JSON.parse(savedScore));
    }
  }, []);

  // Save score to local storage whenever it changes
  useEffect(() => {
    localStorage.setItem('rpsScore', JSON.stringify(score));
  }, [score]);

  const playGame = (choice) => {
    setUserChoice(choice);
    const randomChoice = CHOICES[Math.floor(Math.random() * CHOICES.length)];
    setComputerChoice(randomChoice);
    determineWinner(choice, randomChoice);
  };

  const determineWinner = (user, computer) => {
    if (user === computer) {
      setResult('Draw!');
      setScore(prev => ({ ...prev, draws: prev.draws + 1 }));
    } else if (
      (user === 'rock' && computer === 'scissors') ||
      (user === 'paper' && computer === 'rock') ||
      (user === 'scissors' && computer === 'paper')
    ) {
      setResult('You Win!');
      setScore(prev => ({ ...prev, wins: prev.wins + 1 }));
    } else {
      setResult('You Lose!');
      setScore(prev => ({ ...prev, losses: prev.losses + 1 }));
    }
  };

  const resetGame = () => {
    setUserChoice(null);
    setComputerChoice(null);
    setResult(null);
  };

  const resetScore = () => {
    setScore({ wins: 0, losses: 0, draws: 0 });
    resetGame();
  };

  return (
    <div className="container">
      <h1>Rock Paper Scissors</h1>
      
      <div className="score-board">
        <div className="score-item">Wins: {score.wins}</div>
        <div className="score-item">Losses: {score.losses}</div>
        <div className="score-item">Draws: {score.draws}</div>
      </div>

      <div className="game-area">
        <div className="choices">
          {CHOICES.map((choice) => (
            <button
              key={choice}
              onClick={() => playGame(choice)}
              className="choice-btn"
              disabled={userChoice !== null}
            >
              {choice.toUpperCase()}
            </button>
          ))}
        </div>

        {userChoice && (
          <div className="result-area">
            <div className="result-display">
              <p>You chose: <strong>{userChoice}</strong></p>
              <p>Computer chose: <strong>{computerChoice}</strong></p>
              <h2>{result}</h2>
            </div>
            <button onClick={resetGame} className="play-again-btn">
              Play Again
            </button>
          </div>
        )}
      </div>

      <button onClick={resetScore} className="reset-score-btn">
        Reset Score
      </button>
    </div>
  )
}

export default App
