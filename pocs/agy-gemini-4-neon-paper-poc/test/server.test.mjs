import test from 'node:test';
import assert from 'node:assert/strict';
import { createAppServer } from '../server.mjs';

test('HTTP server serves app shell, health, editions API, Three.js module and blocks traversal', async () => {
  const server = createAppServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;

  try {
    const healthRes = await fetch(`${base}/api/health`);
    assert.equal(healthRes.status, 200);
    const health = await healthRes.json();
    assert.equal(health.status, 'ok');

    const editionsRes = await fetch(`${base}/api/editions`);
    assert.equal(editionsRes.status, 200);
    const editionsBody = await editionsRes.json();
    assert.ok(Array.isArray(editionsBody.editions));
    assert.equal(editionsBody.editions.length, 3);

    const homeRes = await fetch(`${base}/`);
    assert.equal(homeRes.status, 200);
    const html = await homeRes.text();
    assert.ok(html.includes('The Chronicle Gazette'));

    const threeRes = await fetch(`${base}/vendor/three/build/three.module.js`);
    assert.equal(threeRes.status, 200);
    const threeCode = await threeRes.text();
    assert.ok(threeCode.length > 10000);

    const traversalRes = await fetch(`${base}/%2e%2e/package.json`);
    assert.ok(traversalRes.status === 403 || traversalRes.status === 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
