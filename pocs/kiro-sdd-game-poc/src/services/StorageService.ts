import { ScoreState } from '../types';

export class StorageService {
  private static readonly STORAGE_KEY = 'rps-scores';
  private inMemoryStorage: ScoreState = { wins: 0, losses: 0, ties: 0 };
  private useInMemory = false;

  constructor() {
    // Test if localStorage is available
    try {
      const testKey = '__localStorage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
    } catch (e) {
      this.useInMemory = true;
    }
  }

  getScores(): ScoreState {
    if (this.useInMemory) {
      return { ...this.inMemoryStorage };
    }

    try {
      const stored = localStorage.getItem(StorageService.STORAGE_KEY);
      if (!stored) {
        return { wins: 0, losses: 0, ties: 0 };
      }

      const parsed = JSON.parse(stored);
      
      // Validate the structure
      if (
        typeof parsed === 'object' &&
        typeof parsed.wins === 'number' &&
        typeof parsed.losses === 'number' &&
        typeof parsed.ties === 'number'
      ) {
        return parsed;
      } else {
        // Invalid data, return defaults
        return { wins: 0, losses: 0, ties: 0 };
      }
    } catch (e) {
      // Parsing error or other issue, return defaults
      return { wins: 0, losses: 0, ties: 0 };
    }
  }

  saveScores(scores: ScoreState): void {
    if (this.useInMemory) {
      this.inMemoryStorage = { ...scores };
      return;
    }

    try {
      localStorage.setItem(StorageService.STORAGE_KEY, JSON.stringify(scores));
    } catch (e) {
      // If localStorage fails, fall back to in-memory storage
      this.useInMemory = true;
      this.inMemoryStorage = { ...scores };
    }
  }

  resetScores(): void {
    const defaultScores = { wins: 0, losses: 0, ties: 0 };
    
    if (this.useInMemory) {
      this.inMemoryStorage = { ...defaultScores };
      return;
    }

    try {
      localStorage.setItem(StorageService.STORAGE_KEY, JSON.stringify(defaultScores));
    } catch (e) {
      // If localStorage fails, fall back to in-memory storage
      this.useInMemory = true;
      this.inMemoryStorage = { ...defaultScores };
    }
  }

  // Method to check if using in-memory storage (useful for testing)
  isUsingInMemoryStorage(): boolean {
    return this.useInMemory;
  }
}