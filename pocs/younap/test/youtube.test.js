import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseResults, toSeconds } from '../server/youtube.js';

function video(videoId, title, channel, length) {
  const renderer = { videoId, title: { runs: [{ text: title }] }, ownerText: { runs: [{ text: channel }] } };
  if (length) renderer.lengthText = { simpleText: length };
  return { videoRenderer: renderer };
}

function page(contents) {
  const data = { contents: { sectionListRenderer: { contents } } };
  return `<html><script>var ytInitialData = ${JSON.stringify(data)};</script></html>`;
}

test('a duration label becomes seconds so the playlist can sum total time', () => {
  assert.equal(toSeconds('4:02'), 242);
  assert.equal(toSeconds('1:02:03'), 3723);
  assert.equal(toSeconds(''), 0);
});

test('every playable video on the results page becomes a track', () => {
  const tracks = parseResults(page([video('abc', 'Around The World', 'Daft Punk', '4:02')]));
  assert.deepEqual(tracks, [
    { id: 'abc', title: 'Around The World', channel: 'Daft Punk', duration: '4:02', seconds: 242 }
  ]);
});

test('videos nested inside shelves are found because YouTube moves them around', () => {
  const shelf = { shelfRenderer: { content: { verticalListRenderer: { items: [video('deep', 'Nested', 'Chan', '3:00')] } } } };
  const tracks = parseResults(page([video('top', 'Top', 'Chan', '2:00'), shelf]));
  assert.deepEqual(tracks.map(track => track.id), ['top', 'deep']);
});

test('live streams and channels are dropped because a player needs a finite song', () => {
  const tracks = parseResults(page([
    video('live', 'Live Radio', 'Chan'),
    { channelRenderer: { channelId: 'UC1' } },
    video('song', 'Song', 'Chan', '3:10')
  ]));
  assert.deepEqual(tracks.map(track => track.id), ['song']);
});

test('a page without results data fails loud instead of returning an empty list', () => {
  assert.throws(() => parseResults('<html>consent wall</html>'), /ytInitialData/);
});
