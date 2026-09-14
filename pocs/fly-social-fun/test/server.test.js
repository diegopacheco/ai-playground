import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server/server.js';
import { createSimulation } from '../server/simulation.js';

let app;
let base;

before(async () => {
  app = createApp({ simulation: createSimulation({ seed: 3, actChance: 1 }), tickMs: 40 });
  base = `http://localhost:${await app.listen(0)}`;
});

after(() => app.close());

async function readEvents(count) {
  const controller = new AbortController();
  const res = await fetch(`${base}/api/stream`, { signal: controller.signal });
  assert.equal(res.headers.get('content-type'), 'text/event-stream');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const events = [];
  let buffer = '';
  while (events.length < count) {
    const { value } = await reader.read();
    buffer += decoder.decode(value);
    const frames = buffer.split('\n\n');
    buffer = frames.pop();
    for (const frame of frames) if (frame.startsWith('data: ')) events.push(JSON.parse(frame.slice(6)));
  }
  controller.abort();
  return events;
}

test('the stream opens with a full snapshot so a browser can draw the kitchen right away', async () => {
  const [hello] = await readEvents(1);
  assert.equal(hello.type, 'hello');
  assert.equal(hello.tickMs, 40);
  assert.ok(hello.spots.length > 0 && hello.flies.length > 0);
  assert.ok(hello.species.house);
});

test('the stream keeps pushing live buzzes after the snapshot', async () => {
  const events = await readEvents(40);
  assert.ok(events.some((event) => event.type === 'buzz'), 'no buzz arrived on the stream');
  assert.ok(events.some((event) => event.type === 'tick'));
});

test('a fly profile shows its latest buzzes and a missing fly is a 404', async () => {
  const { flies } = await (await fetch(`${base}/api/state`)).json();
  const profile = await (await fetch(`${base}/api/flies/${flies[0].id}`)).json();
  assert.equal(profile.handle, flies[0].handle);
  assert.ok(Array.isArray(profile.buzzes));
  assert.equal((await fetch(`${base}/api/flies/fly-999999`)).status, 404);
});

test('the swatter and snack endpoints act on real spots and reject bad input', async () => {
  const post = (path, body) => fetch(`${base}${path}`, { method: 'POST', body });
  const swat = await post('/api/swat', JSON.stringify({ spot: 'banana' }));
  assert.equal(swat.status, 202);
  assert.equal((await swat.json()).spot, 'banana');
  assert.equal((await post('/api/swat', JSON.stringify({ spot: 'moon' }))).status, 400);
  assert.equal((await post('/api/snack', JSON.stringify({ spot: 'lamp' }))).status, 400);
  assert.equal((await post('/api/snack', '{not json')).status, 400);
});

test('the page and three.js are served but files outside public are not', async () => {
  const page = await fetch(`${base}/`);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /importmap/);
  assert.equal((await fetch(`${base}/vendor/three/three.module.js`)).status, 200);
  assert.equal((await fetch(`${base}/%2e%2e%2fpackage.json`)).status, 404);
  assert.equal((await fetch(`${base}/vendor/three/%2e%2e%2f%2e%2e%2f%2e%2e%2fpackage.json`)).status, 404);
});
