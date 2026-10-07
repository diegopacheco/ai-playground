import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createApp } from '../server/server.js';

async function withApp(searchFn, run) {
  const app = createApp(searchFn).listen(0);
  await once(app, 'listening');
  try {
    await run(`http://localhost:${app.address().port}`);
  } finally {
    app.close();
  }
}

test('search passes the trimmed query to YouTube and returns its tracks', async () => {
  const seen = [];
  const tracks = [{ id: 'abc', title: 'Song', channel: 'Chan', duration: '3:00', seconds: 180 }];
  await withApp(async query => (seen.push(query), tracks), async base => {
    const response = await fetch(`${base}/api/search?q=%20daft%20punk%20`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), tracks);
    assert.deepEqual(seen, ['daft punk']);
  });
});

test('search without a query is rejected before YouTube is called', async () => {
  let called = false;
  await withApp(async () => (called = true, []), async base => {
    const response = await fetch(`${base}/api/search?q=%20`);
    assert.equal(response.status, 400);
    assert.equal(called, false);
  });
});

test('a YouTube failure reaches the UI as a 502 with the reason', async () => {
  await withApp(async () => { throw new Error('YouTube answered 429'); }, async base => {
    const response = await fetch(`${base}/api/search?q=x`);
    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), { error: 'YouTube answered 429' });
  });
});

test('the player page and its modules are served with types the browser accepts', async () => {
  await withApp(async () => [], async base => {
    const page = await fetch(base);
    assert.match(page.headers.get('content-type'), /text\/html/);
    assert.match(await page.text(), /YOUNAP/);
    const module = await fetch(`${base}/js/app.js`);
    assert.match(module.headers.get('content-type'), /text\/javascript/);
  });
});

test('files outside the public folder are never served', async () => {
  await withApp(async () => [], async base => {
    const response = await fetch(`${base}/..%2Fpackage.json`);
    assert.notEqual(response.status, 200);
  });
});
