import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderBuzzHtml, escapeHtml, lifePercent, avatarSvg } from '../public/js/format.js';

test('buzz text can never inject markup into the feed', () => {
  const html = renderBuzzHtml('<img src=x onerror=alert(1)> @evil #pwn');
  assert.ok(!html.includes('<img'));
  assert.match(html, /&lt;img/);
});

test('mentions and hashtags are highlighted so threads and trends are visible', () => {
  const html = renderBuzzHtml('@zzzara brb flying there #BananaLife');
  assert.match(html, /<span class="mention">@zzzara<\/span>/);
  assert.match(html, /<span class="tag">#BananaLife<\/span>/);
});

test('escaped apostrophes are not mistaken for hashtags', () => {
  const html = renderBuzzHtml("it's #YOLO");
  assert.ok(html.includes('it&#39;s'), html);
  assert.equal((html.match(/class="tag"/g) || []).length, 1);
});

test('the life bar stays inside 0 to 100 even for flies past their lifespan', () => {
  assert.equal(lifePercent(40, 28), 100);
  assert.equal(lifePercent(14, 28), 50);
  assert.equal(lifePercent(3, 0), 100);
});

test('avatar colors cannot break out of the svg attributes', () => {
  assert.ok(!avatarSvg({ body: '"/><script>', eyes: '#fff' }).includes('<script>'));
  assert.equal(escapeHtml('"'), '&quot;');
});
