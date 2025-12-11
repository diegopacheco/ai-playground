import { describe, it, beforeEach, afterEach } from 'bun:test';
import * as fc from 'fast-check';
import { StorageService } from '../../services/StorageService';
import { ScoreState } from '../../types';

// Arbitrary generator for valid score states
const scoreStateArbitrary = fc.record({
  wins: fc.nat({ max: 10000 }),
  losses: fc.nat({ max: 10000 }),
  ties: fc.nat({ max: 10000 })
}) as fc.Arbitrary<ScoreState>;

describe('Storage Service Properties', () => {
  let storageService: StorageService;
  let originalLocalStorage: Storage;

  beforeEach(() => {
    // Save original localStorage
    originalLocalStorage = global.localStorage;
    
    // Create a fresh storage service for each test
    storageService = new StorageService();
    
    // Clear any existing data
    storageService.resetScores();
  });

  afterEach(() => {
    // Restore original localStorage
    global.localStorage = originalLocalStorage;
  });

  /**
   * Feature: rock-paper-scissors-game, Property 6: Score persistence round-trip
   * For any game round outcome, storing the updated score to localStorage and then 
   * retrieving it should yield the same score values
   * Validates: Requirements 3.1, 3.2
   */
  it('should maintain score consistency through save and retrieve operations', () => {
    fc.assert(
      fc.property(scoreStateArbitrary, (originalScores) => {
        // Save the scores
        storageService.saveScores(originalScores);
        
        // Retrieve the scores
        const retrievedScores = storageService.getScores();
        
        // They should be identical
        return (
          retrievedScores.wins === originalScores.wins &&
          retrievedScores.losses === originalScores.losses &&
          retrievedScores.ties === originalScores.ties
        );
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Feature: rock-paper-scissors-game, Property 7: Score reset completeness
   * For any score state, when reset is triggered, both the displayed scores and 
   * localStorage should show zero for wins, losses, and ties
   * Validates: Requirements 3.5
   */
  it('should reset all scores to zero regardless of previous state', () => {
    fc.assert(
      fc.property(scoreStateArbitrary, (initialScores) => {
        // Set some initial scores (could be anything)
        storageService.saveScores(initialScores);
        
        // Reset the scores
        storageService.resetScores();
        
        // Retrieve the scores after reset
        const resetScores = storageService.getScores();
        
        // All should be zero
        return (
          resetScores.wins === 0 &&
          resetScores.losses === 0 &&
          resetScores.ties === 0
        );
      }),
      { numRuns: 100 }
    );
  });
});