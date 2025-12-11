import { describe, it, expect } from 'bun:test';
import * as fc from 'fast-check';

describe('Property-based testing setup', () => {
  it('should verify fast-check is working correctly', () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        return n + 0 === n;
      }),
      { numRuns: 100 }
    );
  });

  it('should verify string concatenation property', () => {
    fc.assert(
      fc.property(fc.string(), fc.string(), (a, b) => {
        return (a + b).length === a.length + b.length;
      }),
      { numRuns: 100 }
    );
  });
});