import React, { useState, useEffect } from 'react';
import { questions } from './questions';
import Question from './Question';
import Timer from './Timer';

const Game = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(21);
  const [showConfetti, setShowConfetti] = useState(false);
  const [skipCount, setSkipCount] = useState(2);
  const [usedVibeCode, setUsedVibeCode] = useState(false);

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
      // Game over
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
      const win = Math.random() > 0.7; // 30% chance to win
      if (win) {
        setTimeLeft(timeLeft + 10);
      }
    }
  };

  useEffect(() => {
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
  }, [currentQuestionIndex]);

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
          Vibe Code
        </button>
      </div>
      <div>
        <p>Score: {score}</p>
      </div>
    </div>
  );
};

export default Game;
