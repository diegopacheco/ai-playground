import { useState, useEffect } from 'react';

export type GameResult = {
  id: string;
  playerMove: string;
  computerMove: string;
  result: string;
  timestamp: number;
};

export const useGameHistory = () => {
  const [history, setHistory] = useState<GameResult[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('game-history');
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  const addResult = (result: GameResult) => {
    const newHistory = [result, ...history].slice(0, 50);
    setHistory(newHistory);
    localStorage.setItem('game-history', JSON.stringify(newHistory));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('game-history');
  };

  return { history, addResult, clearHistory };
};
