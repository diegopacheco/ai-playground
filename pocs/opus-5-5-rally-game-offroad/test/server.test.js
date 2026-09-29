import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolvePath } from '../server.js';

test('the game page and its modules are served from public', () => {
  assert.match(resolvePath('/'), /public[\\/]index\.html$/);
  assert.match(resolvePath('/js/main.js'), /public[\\/]js[\\/]main\.js$/);
});

test('three.js is served from node_modules so the game runs offline', () => {
  assert.match(resolvePath('/vendor/three/build/three.module.js'), /node_modules[\\/]three[\\/]build[\\/]three\.module\.js$/);
});

test('path traversal outside the served folders is refused', () => {
  assert.equal(resolvePath('/../package.json'), resolvePath('/package.json'));
  assert.match(resolvePath('/%2e%2e/%2e%2e/etc/passwd'), /public[\\/]etc[\\/]passwd$/);
  assert.equal(resolvePath('/vendor/three/..%2f..%2fserver.js'), null);
});
