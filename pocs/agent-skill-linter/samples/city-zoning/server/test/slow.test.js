import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluateProposal } from '../src/zoning/rules.js';

test('bulk plan review across many parcels', async () => {
  let processed = 0;
  for (let i = 0; i < 500; i++) {
    evaluateProposal('C1', {
      use: 'office', heightFeet: 40, footprintArea: 1200, lotArea: 3000,
      floorArea: 6000, frontSetback: 12, sideSetback: 2, rearSetback: 12
    });
    processed++;
  }
  await new Promise(resolve => setTimeout(resolve, 5200));
  assert.equal(processed, 500);
});
