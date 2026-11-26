import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EDITIONS,
  cloneEdition,
  getEditionById,
  applyHeadlineEdit,
  hitTestArticle,
  validateEdition
} from '../public/js/editions.js';

test('every bundled edition passes structural validation with 8 pages and valid hitboxes', () => {
  assert.ok(EDITIONS.length >= 3);
  for (const edition of EDITIONS) {
    const errors = validateEdition(edition);
    assert.deepEqual(errors, [], `edition ${edition.id} had errors: ${errors.join(', ')}`);
  }
});

test('applyHeadlineEdit updates front page without mutating original edition', () => {
  const original = getEditionById('chronicle-1924');
  const beforeBanner = original.pages[0].banner;
  const beforeMasthead = original.masthead;

  const updated = applyHeadlineEdit(original, {
    masthead: 'The Midnight Press',
    banner: 'City Bridges Illuminated by Arc Lamps',
    deck: 'Crowds gather along the riverfront to witness the first electric skyline.',
    leadBody: 'First paragraph from the night desk.\n\nSecond paragraph confirming full brightness.'
  });

  assert.equal(original.masthead, beforeMasthead);
  assert.equal(original.pages[0].banner, beforeBanner);
  assert.equal(updated.masthead, 'THE MIDNIGHT PRESS');
  assert.equal(updated.pages[0].banner, 'CITY BRIDGES ILLUMINATED BY ARC LAMPS');
  assert.equal(updated.pages[0].articles[0].paragraphs.length, 2);
});

test('hitTestArticle resolves the exact column under normalized UV coordinates', () => {
  const front = getEditionById('chronicle-1924').pages[0];
  const lead = hitTestArticle(front, 0.15, 0.50);
  assert.ok(lead);
  assert.equal(lead.id, 'p1-lead');

  const wireless = hitTestArticle(front, 0.45, 0.75);
  assert.ok(wireless);
  assert.equal(wireless.id, 'p1-wireless');

  const mastheadMiss = hitTestArticle(front, 0.50, 0.08);
  assert.equal(mastheadMiss, null);
});

test('validateEdition rejects editions with missing pages or out-of-bounds boxes', () => {
  const broken = cloneEdition(EDITIONS[0]);
  broken.pages.pop();
  const missingErrs = validateEdition(broken);
  assert.ok(missingErrs.some((e) => e.includes('8 pages')));

  const badBox = cloneEdition(EDITIONS[0]);
  badBox.pages[0].articles[0].box.w = 1.5;
  const boxErrs = validateEdition(badBox);
  assert.ok(boxErrs.some((e) => e.includes('out of bounds')));
});
