import test from 'node:test';
import assert from 'node:assert/strict';
import { createAppServer } from '../server.mjs';

async function withServer(fn) {
  const server = createAppServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  const baseUrl = `http://127.0.0.1:${port}`;
  try {
    await fn(baseUrl);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test('health endpoint confirms service availability', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/api/health`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.deepEqual(body, {
      status: 'ok',
      service: 'colorado-summer-horse'
    });
  });
});

test('presets endpoint exposes coats, gaits, lighting moods, and cameras', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/api/presets`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.coats.length, 4);
    assert.equal(body.gaits.length, 4);
    assert.equal(body.lighting.length, 3);
    assert.equal(body.cameras.length, 5);
  });
});

test('html shell and local three.js vendor module load without external cdn', async () => {
  await withServer(async (baseUrl) => {
    const htmlRes = await fetch(`${baseUrl}/`);
    assert.equal(htmlRes.status, 200);
    const html = await htmlRes.text();
    assert.match(html, /\/vendor\/three\/build\/three\.module\.js/);
    assert.match(html, /\/js\/app\.js/);

    const threeRes = await fetch(`${baseUrl}/vendor/three/build/three.module.js`);
    assert.equal(threeRes.status, 200);
    const js = await threeRes.text();
    assert.ok(js.length > 50000);
  });
});

test('unsupported methods are rejected with 405', async () => {
  await withServer(async (baseUrl) => {
    const res = await fetch(`${baseUrl}/api/health`, { method: 'POST' });
    assert.equal(res.status, 405);
  });
});
