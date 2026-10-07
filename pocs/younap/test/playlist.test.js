import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatTime, nextIndex, previousIndex, removeTrack, scrollText, totalSeconds } from '../public/js/playlist.js';

const tracks = [
  { id: 'a', seconds: 60 },
  { id: 'b', seconds: 125 },
  { id: 'c', seconds: 5 }
];

test('the clock always shows two digit minutes and seconds like the Winamp display', () => {
  assert.equal(formatTime(0), '00:00');
  assert.equal(formatTime(65.9), '01:05');
  assert.equal(formatTime(3723), '62:03');
  assert.equal(formatTime(undefined), '00:00');
});

test('the playlist footer sums every track length', () => {
  assert.equal(totalSeconds(tracks), 190);
  assert.equal(totalSeconds([]), 0);
});

test('playback walks the playlist in order', () => {
  assert.equal(nextIndex(0, 3, { shuffle: false, repeat: false }), 1);
  assert.equal(nextIndex(-1, 3, { shuffle: false, repeat: false }), 0);
});

test('the last song stops playback unless repeat is on', () => {
  assert.equal(nextIndex(2, 3, { shuffle: false, repeat: false }), -1);
  assert.equal(nextIndex(2, 3, { shuffle: false, repeat: true }), 0);
});

test('shuffle never replays the song that just ended', () => {
  for (const random of [0, 0.3, 0.5, 0.99]) {
    const picked = nextIndex(1, 3, { shuffle: true, repeat: false }, () => random);
    assert.notEqual(picked, 1);
    assert.ok(picked >= 0 && picked < 3);
  }
});

test('shuffle with nothing playing can start on any song', () => {
  assert.equal(nextIndex(-1, 3, { shuffle: true, repeat: false }, () => 0), 0);
  assert.equal(nextIndex(-1, 3, { shuffle: true, repeat: false }, () => 0.99), 2);
});

test('an empty playlist has nothing to play in either direction', () => {
  assert.equal(nextIndex(-1, 0, { shuffle: true, repeat: true }), -1);
  assert.equal(previousIndex(-1, 0), -1);
});

test('previous wraps from the first song to the last', () => {
  assert.equal(previousIndex(2, 3), 1);
  assert.equal(previousIndex(0, 3), 2);
});

test('removing a song above the playing one keeps the same song marked as playing', () => {
  const result = removeTrack(tracks, 2, 0);
  assert.deepEqual(result.tracks.map(track => track.id), ['b', 'c']);
  assert.equal(result.tracks[result.current].id, 'c');
});

test('removing a song below the playing one leaves the playing index alone', () => {
  assert.equal(removeTrack(tracks, 0, 2).current, 0);
});

test('removing the playing song leaves nothing marked as playing', () => {
  const result = removeTrack(tracks, 1, 1);
  assert.equal(result.current, -1);
  assert.equal(result.tracks.length, 2);
});

test('removing with no selection changes nothing', () => {
  assert.deepEqual(removeTrack(tracks, 1, -1), { tracks, current: 1 });
});

test('a short title is shown still and a long one scrolls and wraps around', () => {
  assert.equal(scrollText('SHORT', 7, 10), 'SHORT');
  assert.equal(scrollText('ABCDEFGHIJ', 0, 4), 'ABCD');
  assert.equal(scrollText('ABCDEFGHIJ', 8, 6), 'IJ  **');
  assert.equal(scrollText('ABCDEFGHIJ', 17, 4), scrollText('ABCDEFGHIJ', 0, 4));
});
