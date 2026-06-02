import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluateProposal } from '../src/zoning/rules.js';
import { permitFee } from '../src/zoning/permits.js';

test('compliant single family in R1', () => {
  const result = evaluateProposal('R1', {
    use: 'single_family', heightFeet: 30, footprintArea: 1500, lotArea: 6000,
    floorArea: 2500, frontSetback: 30, sideSetback: 10, rearSetback: 30
  });
  assert.equal(result.compliant, true);
  assert.equal(result.violations.length, 0);
});

test('flags use not permitted in zone', () => {
  const result = evaluateProposal('R1', {
    use: 'warehouse', heightFeet: 30, footprintArea: 1500, lotArea: 6000,
    floorArea: 2500, frontSetback: 30, sideSetback: 10, rearSetback: 30
  });
  assert.equal(result.compliant, false);
  assert.ok(result.violations.some(v => v.rule === 'permitted_use'));
});

test('flags height and lot coverage violations', () => {
  const result = evaluateProposal('R1', {
    use: 'single_family', heightFeet: 50, footprintArea: 4000, lotArea: 6000,
    floorArea: 2500, frontSetback: 30, sideSetback: 10, rearSetback: 30
  });
  assert.ok(result.violations.some(v => v.rule === 'max_height'));
  assert.ok(result.violations.some(v => v.rule === 'lot_coverage'));
});

test('unknown zone throws', () => {
  assert.throws(() => evaluateProposal('ZZ', {}));
});

test('permit fee is zero when not compliant', () => {
  assert.equal(permitFee({ floorArea: 3000, use: 'retail', heightFeet: 20 }, false), 0);
});

test('permit fee scales with floor area, use, and height', () => {
  const fee = permitFee({ floorArea: 12000, use: 'manufacturing', heightFeet: 70 }, true);
  assert.equal(fee, 250 + 1500 + 500 + 400);
});
