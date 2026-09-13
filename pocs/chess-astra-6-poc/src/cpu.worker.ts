import { chooseMove, restoreGame, type SearchRequest } from './engine';

self.onmessage = ({ data }: MessageEvent<SearchRequest>) => {
  try {
    self.postMessage({ move: chooseMove(restoreGame(data.history), data.difficulty) });
  } catch {
    self.postMessage({ error: 'The guardian could not calculate a move. Undo or start a new game.' });
  }
};
