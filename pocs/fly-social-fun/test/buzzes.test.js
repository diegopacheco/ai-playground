import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TEMPLATES, fill, writeBuzz } from '../server/buzzes.js';
import { extractHashtags, trendingHashtags } from '../server/social.js';
import { NAMES, SNACKS, SPOTS } from '../server/world.js';
import { createRandom } from '../server/random.js';

const LONGEST_HANDLE = NAMES.map(([, handle]) => `${handle}_10`).sort((a, b) => b.length - a.length)[0];
const LONGEST_SPOT = SPOTS.map((spot) => spot.label.toUpperCase()).sort((a, b) => b.length - a.length)[0];
const LONGEST_SNACK = SNACKS.map((snack) => snack.toUpperCase()).sort((a, b) => b.length - a.length)[0];
const WORST_CASE = { handle: LONGEST_HANDLE, parent: LONGEST_HANDLE, spot: LONGEST_SPOT, snack: LONGEST_SNACK, age: 40, count: 25 };

test('every buzz fits in a tweet even with the longest names', () => {
  for (const [kind, templates] of Object.entries(TEMPLATES)) {
    for (const template of templates) {
      const text = fill(template, WORST_CASE);
      assert.ok(text.length <= 280, `${kind} is ${text.length} chars: ${text}`);
    }
  }
});

test('replies mention the fly they answer so the thread makes sense', () => {
  const rng = createRandom(3);
  for (let i = 0; i < 30; i++) {
    assert.match(writeBuzz('reply', rng, { handle: 'zzzara' }), /@zzzara\b/);
  }
});

test('a template missing a value fails loud instead of posting a broken buzz', () => {
  assert.throws(() => fill('RIP @{handle}', {}), /missing value handle/);
  assert.throws(() => writeBuzz('nope', createRandom(1)), /unknown buzz kind/);
});

test('trending ranks the hashtags flies use the most', () => {
  const buzzes = ['a #BananaLife', 'b #BananaLife #YOLO', 'c #YOLO', 'd #BananaLife'].map((text) => ({
    hashtags: extractHashtags(text),
  }));
  assert.deepEqual(trendingHashtags(buzzes, 2), [
    { tag: '#BananaLife', count: 3 },
    { tag: '#YOLO', count: 2 },
  ]);
});

test('a hashtag repeated in one buzz is only counted once', () => {
  assert.deepEqual(extractHashtags('#YOLO #YOLO #yolo'), ['#YOLO', '#yolo']);
});
