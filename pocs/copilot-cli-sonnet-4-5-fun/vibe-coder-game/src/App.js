import React, { useState, useEffect, useCallback } from 'react';
import confetti from 'canvas-confetti';
import { allQuestions } from './questions';
import './App.css';

const TOTAL_QUESTIONS = 10;
const TIME_PER_QUESTION = 21;
const MAX_SKIPS = 2;

function App() {
  const [gameState, setGameState] = useState('start');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [questionsPool, setQuestionsPool] = useState([]);
  const [answeredCount, setAnsweredCount] = useState(0);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(TIME_PER_QUESTION);
  const [skipsLeft, setSkipsLeft] = useState(MAX_SKIPS);
  const [vibeUsed, setVibeUsed] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);

  const loadProgress = useCallback(() => {
    const saved = localStorage.getItem('vibeCoderProgress');
    if (saved) {
      const data = JSON.parse(saved);
      setScore(data.score || 0);
      setAnsweredCount(data.answeredCount || 0);
      setSkipsLeft(data.skipsLeft ?? MAX_SKIPS);
      setVibeUsed(data.vibeUsed || false);
      setQuestionsPool(data.questionsPool || []);
      setCurrentQuestionIndex(data.currentQuestionIndex || 0);
      if (data.gameState && data.gameState !== 'start') {
        setGameState(data.gameState);
      }
    }
  }, []);

  const saveProgress = useCallback((data) => {
    localStorage.setItem('vibeCoderProgress', JSON.stringify(data));
  }, []);

  const resetGame = useCallback(() => {
    const shuffled = [...allQuestions].sort(() => Math.random() - 0.5);
    const newPool = shuffled.slice(0, TOTAL_QUESTIONS);
    
    const newState = {
      score: 0,
      answeredCount: 0,
      skipsLeft: MAX_SKIPS,
      vibeUsed: false,
      questionsPool: newPool,
      currentQuestionIndex: 0,
      gameState: 'playing'
    };
    
    setScore(0);
    setAnsweredCount(0);
    setSkipsLeft(MAX_SKIPS);
    setVibeUsed(false);
    setQuestionsPool(newPool);
    setCurrentQuestionIndex(0);
    setTimeLeft(TIME_PER_QUESTION);
    setSelectedAnswer(null);
    setShowResult(false);
    setGameState('playing');
    
    saveProgress(newState);
  }, [saveProgress]);

  useEffect(() => {
    loadProgress();
  }, [loadProgress]);

  const moveToNextQuestion = useCallback((counted) => {
    const newAnsweredCount = counted ? answeredCount + 1 : answeredCount;
    
    if (newAnsweredCount >= TOTAL_QUESTIONS) {
      setGameState('finished');
      saveProgress({
        score,
        answeredCount: newAnsweredCount,
        skipsLeft,
        vibeUsed,
        questionsPool,
        currentQuestionIndex: 0,
        gameState: 'finished'
      });
      return;
    }

    const nextIndex = currentQuestionIndex + 1;
    setCurrentQuestionIndex(nextIndex);
    setAnsweredCount(newAnsweredCount);
    setTimeLeft(TIME_PER_QUESTION);
    setSelectedAnswer(null);
    setShowResult(false);
    
    saveProgress({
      score: isCorrect ? score + 1 : score,
      answeredCount: newAnsweredCount,
      skipsLeft,
      vibeUsed,
      questionsPool,
      currentQuestionIndex: nextIndex,
      gameState: 'playing'
    });
  }, [answeredCount, currentQuestionIndex, isCorrect, questionsPool, saveProgress, score, skipsLeft, vibeUsed]);

  const fireConfetti = () => {
    const duration = 3000;
    const end = Date.now() + duration;

    const frame = () => {
      confetti({
        particleCount: 7,
        angle: 60,
        spread: 55,
        origin: { x: 0 },
        colors: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
      });
      confetti({
        particleCount: 7,
        angle: 120,
        spread: 55,
        origin: { x: 1 },
        colors: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
      });

      if (Date.now() < end) {
        requestAnimationFrame(frame);
      }
    };
    frame();
  };

  const handleTimeUp = useCallback(() => {
    setShowResult(true);
    setIsCorrect(false);
    
    setTimeout(() => {
      moveToNextQuestion(false);
    }, 2000);
  }, [moveToNextQuestion]);

  const handleAnswer = (index) => {
    if (showResult) return;
    
    setSelectedAnswer(index);
    const correct = index === questionsPool[currentQuestionIndex].correct;
    setIsCorrect(correct);
    setShowResult(true);

    if (correct) {
      setScore((prev) => prev + 1);
      fireConfetti();
    }

    setTimeout(() => {
      moveToNextQuestion(true);
    }, 3000);
  };

  useEffect(() => {
    if (gameState !== 'playing' || showResult) return;

    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          handleTimeUp();
          return TIME_PER_QUESTION;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gameState, showResult, currentQuestionIndex, handleTimeUp]);

  const handleSkip = () => {
    if (skipsLeft <= 0 || showResult) return;

    const availableQuestions = allQuestions.filter(
      (q) => !questionsPool.includes(q)
    );
    
    if (availableQuestions.length === 0) {
      alert('No more questions available to skip to!');
      return;
    }

    const randomQ = availableQuestions[Math.floor(Math.random() * availableQuestions.length)];
    const newPool = [...questionsPool];
    newPool[currentQuestionIndex] = randomQ;
    
    setQuestionsPool(newPool);
    setSkipsLeft((prev) => prev - 1);
    setTimeLeft(TIME_PER_QUESTION);
    setSelectedAnswer(null);
    setShowResult(false);
    
    saveProgress({
      score,
      answeredCount,
      skipsLeft: skipsLeft - 1,
      vibeUsed,
      questionsPool: newPool,
      currentQuestionIndex,
      gameState: 'playing'
    });
  };

  const handleVibeCoding = () => {
    if (vibeUsed || showResult) return;

    const vibeOptions = [
      { type: 'win', message: '🎉 VIBE CHECK PASSED! +10 seconds!', bonus: 10 },
      { type: 'lose', message: '💀 VIBE CHECK FAILED! No bonus.', bonus: 0 },
      { type: 'lose', message: '😱 COMPILER ERROR! No bonus.', bonus: 0 }
    ];

    const result = vibeOptions[Math.floor(Math.random() * vibeOptions.length)];
    
    if (result.type === 'win') {
      setTimeLeft((prev) => prev + result.bonus);
      alert(result.message);
    } else {
      alert(result.message);
    }

    setVibeUsed(true);
    saveProgress({
      score,
      answeredCount,
      skipsLeft,
      vibeUsed: true,
      questionsPool,
      currentQuestionIndex,
      gameState: 'playing'
    });
  };

  if (gameState === 'start') {
    return (
      <div className="app">
        <div className="start-screen">
          <h1 className="title">💻 Who Wants to be a</h1>
          <h1 className="title-main">VIBE CODER?</h1>
          <p className="subtitle">Think you can handle 10 questions about programming's wildest topics?</p>
          <button className="start-button" onClick={resetGame}>
            START VIBING 🚀
          </button>
          <div className="rules">
            <h3>The Rules:</h3>
            <ul>
              <li>Answer 10 questions correctly</li>
              <li>21 seconds per question</li>
              <li>Skip up to 2 questions (but you still need 10 answers!)</li>
              <li>Use "Vibe Code" once for a chance at +10 seconds</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  if (gameState === 'finished') {
    return (
      <div className="app">
        <div className="finished-screen">
          <h1 className="title">🎊 GAME OVER! 🎊</h1>
          <div className="final-score">
            <h2>Final Score: {score}/{TOTAL_QUESTIONS}</h2>
            <p className="score-message">
              {score === TOTAL_QUESTIONS && "🔥 PERFECT! You're a true Vibe Coder!"}
              {score >= 7 && score < TOTAL_QUESTIONS && "💪 Great job! You've got the vibes!"}
              {score >= 5 && score < 7 && "👍 Not bad! Keep vibing!"}
              {score < 5 && "📚 Time to hit the docs! Keep learning!"}
            </p>
          </div>
          <button className="start-button" onClick={resetGame}>
            PLAY AGAIN 🔄
          </button>
        </div>
      </div>
    );
  }

  const currentQuestion = questionsPool[currentQuestionIndex];

  if (!currentQuestion) {
    return (
      <div className="app">
        <div className="loading">Loading questions...</div>
      </div>
    );
  }

  return (
    <div className="app">
      <div className="game-screen">
        <div className="header">
          <div className="stats">
            <span>Question: {answeredCount + 1}/{TOTAL_QUESTIONS}</span>
            <span>Score: {score}</span>
          </div>
          <div className={`timer ${timeLeft <= 5 ? 'timer-warning' : ''}`}>
            ⏱️ {timeLeft}s
          </div>
        </div>

        <div className="question-container">
          <h2 className="question">{currentQuestion.question}</h2>
          
          {showResult && (
            <div className={`result-message ${isCorrect ? 'correct' : 'incorrect'}`}>
              {isCorrect ? '🎉 CORRECT! VIBE CHECK PASSED!' : '❌ WRONG! BETTER LUCK NEXT TIME!'}
            </div>
          )}

          <div className="options">
            {currentQuestion.options.map((option, index) => (
              <button
                key={index}
                className={`option ${selectedAnswer === index ? (isCorrect ? 'correct' : 'incorrect') : ''} ${showResult && index === currentQuestion.correct ? 'correct' : ''}`}
                onClick={() => handleAnswer(index)}
                disabled={showResult}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

        <div className="lifelines">
          <button
            className="lifeline-btn"
            onClick={handleSkip}
            disabled={skipsLeft <= 0 || showResult}
          >
            ⏭️ SKIP ({skipsLeft} left)
          </button>
          <button
            className="lifeline-btn vibe-btn"
            onClick={handleVibeCoding}
            disabled={vibeUsed || showResult}
          >
            ✨ VIBE CODE {vibeUsed ? '(USED)' : ''}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
