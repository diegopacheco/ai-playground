import React, { useState, useEffect } from 'react';
import { questions } from './questions';
import Question from './Question';
import Timer from './Timer';

const Game = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(() => {
    const savedIndex = localStorage.getItem('currentQuestionIndex');
    return savedIndex !== null ? parseInt(savedIndex, 10) : 0;
  });
  const [score, setScore] = useState(() => {
    const savedScore = localStorage.getItem('score');
    return savedScore !== null ? parseInt(savedScore, 10) : 0;
  });
  const [timeLeft, setTimeLeft] = useState(21);
  const [showConfetti, setShowConfetti] = useState(false);
  const [skipCount, setSkipCount] = useState(() => {
    const savedSkipCount = localStorage.getItem('skipCount');
    return savedSkipCount !== null ? parseInt(savedSkipCount, 10) : 2;
  });
  const [usedVibeCode, setUsedVibeCode] = useState(() => {
    const savedUsedVibeCode = localStorage.getItem('usedVibeCode');
    return savedUsedVibeCode === 'true';
  });
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    localStorage.setItem('currentQuestionIndex', currentQuestionIndex);
    localStorage.setItem('score', score);
    localStorage.setItem('skipCount', skipCount);
    localStorage.setItem('usedVibeCode', usedVibeCode);
  }, [currentQuestionIndex, score, skipCount, usedVibeCode]);

  const handleAnswer = (isCorrect) => {
    if (isCorrect) {
      setScore(score + 1);
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 2000);
    }
    goToNextQuestion();
  };

  const goToNextQuestion = () => {
    setTimeLeft(21);
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      setGameOver(true);
    }
  };

  const handleSkip = () => {
    if (skipCount > 0) {
      setSkipCount(skipCount - 1);
      goToNextQuestion();
    }
  };

  const handleVibeCode = () => {
    if (!usedVibeCode) {
      setUsedVibeCode(true);
      const win = Math.random() > 0.7;
      if (win) {
        setTimeLeft(timeLeft + 10);
      }
    }
  };

  const restartGame = () => {
    localStorage.clear();
    setCurrentQuestionIndex(0);
    setScore(0);
    setTimeLeft(21);
    setSkipCount(2);
    setUsedVibeCode(false);
    setGameOver(false);
  };

  useEffect(() => {
    if (gameOver) return;

    const timer = setInterval(() => {
      setTimeLeft((prevTime) => {
        if (prevTime === 1) {
          goToNextQuestion();
          return 21;
        }
        return prevTime - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [currentQuestionIndex, gameOver]);

  if (gameOver) {
    return (
      <div>
        <h2>Game Over!</h2>
        <p>Your final score: {score} out of {questions.length}</p>
        <button onClick={restartGame}>Play Again</button>
      </div>
    );
  }

  return (
    <div>
      {showConfetti && <div className="confetti">🎉</div>}
      <Timer timeLeft={timeLeft} />
      <Question
        question={questions[currentQuestionIndex]}
        handleAnswer={handleAnswer}
      />
      <div>
        <button onClick={handleSkip} disabled={skipCount === 0}>
          Skip ({skipCount} left)
        </button>
        <button onClick={handleVibeCode} disabled={usedVibeCode}>
          Vibe Code ({usedVibeCode ? 'Used' : 'Available'})
        </button>
      </div>
      <div>
        <p>Score: {score}</p>
      </div>
    </div>
  );
};

export default Game;
