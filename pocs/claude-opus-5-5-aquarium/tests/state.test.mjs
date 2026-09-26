import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MAX_FISH, MAX_GRASS } from '../web/catalog.mjs';
import * as S from '../web/state.mjs';

test('adding fish stops at the tank capacity so the scene stays smooth', () => {
  let s = S.clearFish(S.createState());
  for (let i = 0; i < MAX_FISH + 5; i++) s = S.addFish(s, 'guppy');
  assert.equal(s.fish.length, MAX_FISH);
  assert.equal(S.addFish(s, 'neon'), s);
});

test('removing a species takes out one fish of that species only', () => {
  let s = S.clearFish(S.createState());
  s = S.addFish(S.addFish(S.addFish(s, 'clown'), 'neon'), 'clown');
  s = S.removeFish(s, 'clown');
  assert.equal(S.countFish(s, 'clown'), 1);
  assert.equal(S.countFish(s, 'neon'), 1);
  assert.equal(S.removeFish(s, 'betta'), s);
});

test('decorations toggle on and off without duplicates', () => {
  let s = S.createState();
  const had = s.decor.includes('car');
  s = S.toggleDecor(s, 'car');
  assert.equal(s.decor.includes('car'), !had);
  s = S.toggleDecor(s, 'car');
  assert.equal(s.decor.includes('car'), had);
  assert.equal(new Set(s.decor).size, s.decor.length);
});

test('unknown ids are rejected instead of corrupting the tank', () => {
  const s = S.createState();
  assert.throws(() => S.setMaterial(s, 'cheese'));
  assert.throws(() => S.setRackColor(s, 'plaid'));
  assert.throws(() => S.addFish(s, 'kraken'));
  assert.throws(() => S.toggleDecor(s, 'volcano'));
  assert.throws(() => S.setSubstrate(s, 'lava'));
  assert.throws(() => S.toggleScape(s, 'cactus'));
});

test('state changes return new objects so the renderer can diff them', () => {
  const s = S.createState();
  const next = S.setMaterial(s, 'bricks');
  assert.notEqual(next, s);
  assert.equal(s.material, 'wood');
  assert.equal(next.material, 'bricks');
});

test('restoring a saved tank drops bad entries and never auto-plays sound', () => {
  const s = S.restore({ rackColor: 'cherry', material: 'nope', fish: ['neon', 'kraken'], decor: ['car', 'car', 'moon'], shark: true, sound: true });
  assert.equal(s.rackColor, 'cherry');
  assert.equal(s.material, 'wood');
  assert.deepEqual(s.fish, ['neon']);
  assert.deepEqual(s.decor, ['car']);
  assert.equal(s.shark, true);
  assert.equal(s.sound, false);
  assert.deepEqual(S.restore(null), S.createState());
});

test('the shark and sound switches flip', () => {
  const s = S.createState();
  assert.equal(S.toggleShark(s).shark, !s.shark);
  assert.equal(S.toggleSound(s).sound, !s.sound);
});

test('seagrass can be added and removed one level at a time within limits', () => {
  let s = S.setGrass(S.createState(), 0);
  assert.equal(s.grass, 0);
  assert.equal(S.setGrass(s, -1), s);
  for (let i = 0; i < MAX_GRASS + 3; i++) s = S.setGrass(s, s.grass + 1);
  assert.equal(s.grass, MAX_GRASS);
  assert.equal(S.restore({ grass: 99 }).grass, MAX_GRASS);
  assert.equal(S.restore({ grass: 'lots' }).grass, S.createState().grass);
});

test('the substrate switches and an old save without aquascape fields gets the planted defaults', () => {
  const s = S.setSubstrate(S.createState(), 'sand');
  assert.equal(s.substrate, 'sand');
  const old = S.restore({ rackColor: 'cherry' });
  assert.equal(old.substrate, S.createState().substrate);
  assert.deepEqual(old.scape, S.createState().scape);
  const bad = S.restore({ substrate: 'lava', scape: ['carpet', 'carpet', 'cactus'] });
  assert.equal(bad.substrate, S.createState().substrate);
  assert.deepEqual(bad.scape, ['carpet']);
});

test('aquascape items toggle on and off without duplicates', () => {
  let s = S.createState();
  const had = s.scape.includes('stones');
  s = S.toggleScape(s, 'stones');
  assert.equal(s.scape.includes('stones'), !had);
  s = S.toggleScape(s, 'stones');
  assert.equal(s.scape.includes('stones'), had);
  assert.equal(new Set(s.scape).size, s.scape.length);
});
