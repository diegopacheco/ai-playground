import { test, expect, type Page } from '@playwright/test';

async function move(page: Page, notation: string) {
  const input = page.getByRole('textbox', { name: 'Move notation' });
  await expect(input).toBeEnabled();
  await input.fill(notation);
  await page.getByRole('button', { name: 'Play move', exact: true }).click();
}
async function seed(page: Page, history: string[]) {
  await page.addInitScript(history => localStorage.setItem('wizards-gambit-v1', JSON.stringify({ history, difficulty: 'wizard' })), history);
  await page.goto('/');
}

test('renders the hall, plays against CPU, saves, undoes and restarts', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#scene canvas')).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: 'printscreens/great-hall.png', fullPage: true });
  await move(page, 'e2e4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  await expect(page.locator('#status-title')).toHaveText('The board is yours.');
  await page.reload();
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  await page.getByRole('button', { name: 'Take back a turn' }).click();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await move(page, 'e2e5');
  await expect(page.locator('#status-title')).toHaveText('That spell does not quite work.');
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await page.getByLabel('CHOOSE YOUR CHALLENGE').selectOption('grandmaster');
  await move(page, 'd4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  await page.screenshot({ path: 'printscreens/active-match.png', fullPage: true });
  await page.getByRole('button', { name: 'New game', exact: false }).first().click();
  await page.getByRole('button', { name: 'Keep playing' }).click();
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  await page.locator('#new-game').click();
  await page.locator('#confirm-restart').click();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  expect(errors).toEqual([]);
});

test('board can be clicked, rotated, viewed overhead and muted', async ({ page }) => {
  await page.route(/onlinesequencer/, route => route.abort());
  await page.goto('/');
  await page.getByRole('button', { name: 'Switch to overhead view' }).click();
  await expect(page.getByRole('button', { name: 'Switch to perspective view' })).toBeVisible();
  const canvas = page.locator('#scene canvas');
  await expect(canvas).toBeVisible();
  const rect = (await canvas.boundingBox())!;
  const halfHeight = Math.tan(37 * Math.PI / 360) * 22.78;
  const scale = rect.height / (2 * halfHeight);
  await canvas.click({ position: { x: rect.width / 2 + 0.5 * scale, y: rect.height / 2 + 2.5 * scale } });
  await expect(page.locator('#status-title')).toHaveText('Pawn on e2.');
  await canvas.click({ position: { x: rect.width / 2 + 0.5 * scale, y: rect.height / 2 + 0.5 * scale } });
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  await page.getByRole('button', { name: 'Rotate board' }).click();
  await page.screenshot({ path: 'printscreens/overhead.png', fullPage: true });
  await page.getByRole('button', { name: 'Switch to perspective view' }).click();
  await page.screenshot({ path: 'test-results/reverse-view.png' });
  await expect(page.locator('#scene canvas')).toBeVisible();
  await page.getByRole('button', { name: 'Play library music and sound' }).click();
  await expect(page.getByRole('button', { name: 'Mute library music and sound' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Mute library music and sound' }).click();
});

test('capture animates and CPU responds', async ({ page }) => {
  await seed(page, ['e4', 'd5']);
  await move(page, 'exd5');
  await expect(page.locator('#move-count')).toHaveText('3 MOVES');
  await page.waitForTimeout(200);
  await page.screenshot({ path: 'printscreens/capture.png', fullPage: true });
  await expect(page.locator('#move-count')).toHaveText('4 MOVES', { timeout: 15000 });
});

test('checkmate is announced and further moves are disabled', async ({ page }) => {
  await seed(page, ['f3', 'e5', 'g4', 'Qh4#']);
  await expect(page.locator('#status-title')).toHaveText('The Guardian prevails.');
  await expect(page.getByRole('heading', { name: 'Checkmate', exact: true })).toBeVisible();
  await expect(page.locator('#result-winner')).toContainText('The Castle Guardian wins.');
  await expect(page.locator('#result-new-game')).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeDisabled();
  await page.screenshot({ path: 'printscreens/checkmate.png', fullPage: true });
  await page.getByRole('button', { name: 'Take back a turn' }).click();
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  await expect(page.locator('#match-result')).toBeHidden();
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeEnabled();
});

test('guide, keyboard focus, mobile layout and reduced motion work', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('#scene canvas')).toBeVisible();
  await page.getByRole('button', { name: 'How to play' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.screenshot({ path: 'printscreens/spellbook.png', fullPage: true });
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await page.screenshot({ path: 'printscreens/mobile.png', fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await move(page, 'Nf3');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
});

test('broken storage and missing WebGL retain a playable board', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('wizards-gambit-v1', '{broken');
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...args: unknown[]) {
      if (type === 'webgl2') return null;
      return original.call(this, type as '2d', ...args as []);
    } as typeof original;
  });
  await page.goto('/');
  await expect(page.locator('#fallback-board button')).toHaveCount(64);
  await page.getByRole('radio', { name: 'Black', exact: true }).check();
  await expect(page.locator('#fallback-board [data-square="e2"]')).toHaveCSS('color', 'rgb(32, 39, 42)');
  await expect(page.locator('#fallback-board [data-square="e7"]')).toHaveCSS('color', 'rgb(249, 236, 210)');
  await move(page, 'e4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
});

test('promotion asks for a piece and supports underpromotion', async ({ page }) => {
  await seed(page, ['a4', 'h5', 'a5', 'h4', 'a6', 'h3', 'axb7', 'hxg2']);
  await move(page, 'b7a8');
  await expect(page.locator('#promotion')).toBeVisible();
  await page.screenshot({ path: 'printscreens/promotion.png', fullPage: true });
  await page.getByRole('button', { name: 'Knight', exact: false }).click();
  await expect(page.locator('#history')).toContainText('bxa8=N');
  await expect(page.locator('#move-count')).toHaveText('10 MOVES', { timeout: 15000 });
});

test('undo cancels a pending CPU turn without a late reply', async ({ page }) => {
  await page.goto('/');
  await move(page, 'e4');
  await page.getByRole('button', { name: 'Take back a turn' }).click();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await page.waitForTimeout(2100);
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeEnabled();
});

test('restored repetition ends the match as a draw', async ({ page }) => {
  await seed(page, ['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8']);
  await expect(page.locator('#status-text')).toHaveText('Draw by threefold repetition.');
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeDisabled();
  await expect(page.getByRole('heading', { name: 'Draw', exact: true })).toBeVisible();
  await expect(page.locator('#result-winner')).toHaveText('Draw by threefold repetition. No one wins this duel.');
});

test('stalemate shows a result card so the finished match never looks frozen', async ({ page }) => {
  await seed(page, ['e3', 'a5', 'Qh5', 'Ra6', 'Qxa5', 'h5', 'h4', 'Rah6', 'Qxc7', 'f6', 'Qxd7+', 'Kf7', 'Qxb7', 'Qd3', 'Qxb8', 'Qh7', 'Qxc8', 'Kg6', 'Qe6']);
  await expect(page.locator('#status-text')).toHaveText('Stalemate. There are no legal moves.');
  await expect(page.getByRole('heading', { name: 'Stalemate', exact: true })).toBeVisible();
  await expect(page.locator('#result-new-game')).toBeFocused();
  await page.locator('#result-new-game').click();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await expect(page.locator('#match-result')).toBeHidden();
});

test('the page stays fixed across desktop, mobile and landscape viewports', async ({ page }) => {
  await page.goto('/');
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1366, height: 768 }, { width: 1024, height: 768 }, { width: 768, height: 1024 }, { width: 390, height: 844 }, { width: 375, height: 667 }, { width: 320, height: 568 }, { width: 844, height: 390 }]) {
    await page.setViewportSize(viewport);
    await expect(page.locator('#scene canvas')).toBeVisible();
    await page.locator('#new-game').scrollIntoViewIfNeeded();
    await page.locator('#move-form').scrollIntoViewIfNeeded();
    const bounds = await page.evaluate(() => ({
      height: document.documentElement.scrollHeight,
      width: document.documentElement.scrollWidth,
      top: window.scrollY,
      viewportHeight: innerHeight,
      viewportWidth: innerWidth,
      canvasHeight: document.querySelector('canvas')!.getBoundingClientRect().height,
    }));
    expect(bounds.height, JSON.stringify(viewport)).toBeLessThanOrEqual(bounds.viewportHeight);
    expect(bounds.width, JSON.stringify(viewport)).toBeLessThanOrEqual(bounds.viewportWidth);
    expect(bounds.top).toBe(0);
    expect(bounds.canvasHeight).toBeGreaterThan(100);
    if (viewport.width === 1366) await page.screenshot({ path: 'printscreens/laptop.png', fullPage: true });
  }
});

test('the sequencer stays until the player clicks into it, ignores its own focus grab, and unloads when muted', async ({ page }) => {
  await page.route(/onlinesequencer/, () => {});
  await page.goto('/');
  const panel = page.locator('#music-panel');
  const player = page.locator('#music-player');
  await expect(panel).toBeHidden();
  await page.getByRole('button', { name: 'Play library music and sound' }).click();
  await expect(panel).toBeInViewport();
  await expect(player).toHaveAttribute('src', 'https://onlinesequencer.net/1073884');
  const focusSequencer = () => page.evaluate(() => { document.getElementById('music-player')!.focus(); window.dispatchEvent(new Event('blur')); });
  await focusSequencer();
  await expect(panel).toBeInViewport();
  await page.locator('#music-player').dispatchEvent('load');
  await expect(page.locator('#sound')).toBeFocused();
  await focusSequencer();
  await expect(panel).not.toBeInViewport();
  await expect(player).toHaveAttribute('src', 'https://onlinesequencer.net/1073884');
  await expect(page.locator('#music-status')).toHaveText('HEDWIG’S THEME · NOW PLAYING');
  await page.getByRole('button', { name: 'Mute library music and sound' }).click();
  await expect(panel).toBeHidden();
  await expect(player).toHaveAttribute('src', 'about:blank');
  await page.getByRole('button', { name: 'Play library music and sound' }).click();
  await expect(panel).toBeInViewport();
});

test('a capture and a fallen king play death sounds timed to their animations', async ({ page }) => {
  await page.addInitScript(() => {
    const sounds = { tones: [] as number[], noises: [] as number[] };
    Object.assign(window, { soundCheck: sounds });
    const delay = (context: BaseAudioContext, when: number) => Math.round((when - context.currentTime) * 10) / 10;
    const tone = OscillatorNode.prototype.start;
    OscillatorNode.prototype.start = function (when = 0) {
      sounds.tones.push(delay(this.context, when));
      return tone.call(this, when);
    };
    const source = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (when = 0, ...args) {
      if (!this.loop) sounds.noises.push(delay(this.context, when));
      return source.call(this, when, ...args);
    };
  });
  await page.route(/onlinesequencer/, route => route.abort());
  await seed(page, ['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6']);
  await page.getByRole('button', { name: 'Play library music and sound' }).click();
  await expect(page.getByRole('button', { name: 'Mute library music and sound' })).toBeVisible();
  const sounds = () => page.evaluate(() => (window as unknown as { soundCheck: { tones: number[]; noises: number[] } }).soundCheck);
  await move(page, 'd3');
  await expect.poll(async () => (await sounds()).tones.length).toBeGreaterThan(0);
  expect((await sounds()).noises).toEqual([]);
  await expect(page.locator('#move-count')).toHaveText('8 MOVES', { timeout: 15000 });
  await page.getByRole('button', { name: 'Take back a turn' }).click();
  await page.evaluate(() => Object.assign((window as unknown as { soundCheck: object }).soundCheck, { tones: [], noises: [] }));
  await move(page, 'Qxf7#');
  await expect(page.getByRole('heading', { name: 'Checkmate', exact: true })).toBeVisible();
  await expect.poll(async () => (await sounds()).noises.length).toBe(3);
  const { tones, noises } = await sounds();
  expect(noises.sort()).toEqual([0, 0, 0.9]);
  expect(tones).toContain(0.9);
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'printscreens/king-falls.png', fullPage: true });
});

test('piece colors update immediately, survive reload and keep the match intact', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/');
  for (const color of ['White', 'Green', 'Brown', 'Black', 'Blue', 'Orange', 'Salmon', 'Gray']) {
    await page.getByRole('radio', { name: color, exact: true }).check();
    await expect(page.getByRole('radio', { name: color, exact: true })).toBeChecked();
    await expect(page.locator('#human-heading')).toHaveText(color.toUpperCase());
    await expect(page.locator('#cpu-heading')).toHaveText(color === 'White' ? 'GREEN' : 'WHITE');
    await expect(page.locator('#move-count')).toHaveText('0 MOVES');
    await page.screenshot({ path: `printscreens/pieces-${color.toLowerCase()}.png`, fullPage: true });
  }
  await move(page, 'e4');
  await page.getByRole('radio', { name: 'Brown', exact: true }).check();
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  await page.reload();
  await expect(page.getByRole('radio', { name: 'Brown', exact: true })).toBeChecked();
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  await page.getByRole('button', { name: 'Take back a turn' }).click();
  await expect(page.getByRole('radio', { name: 'Brown', exact: true })).toBeChecked();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await page.locator('#new-game').click();
  await expect(page.getByRole('radio', { name: 'Brown', exact: true })).toBeChecked();
});

test('game fullscreen contains the board and controls and can be exited', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Enter game fullscreen' }).click();
  await expect.poll(() => page.evaluate(() => document.fullscreenElement?.id)).toBe('game-layout');
  await expect(page.getByRole('button', { name: 'Exit game fullscreen' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#scene canvas')).toBeInViewport();
  await expect(page.getByRole('radio', { name: 'Green', exact: true })).toBeInViewport();
  await page.getByRole('radio', { name: 'Green', exact: true }).check();
  await move(page, 'e4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  await page.screenshot({ path: 'printscreens/fullscreen.png', fullPage: true });
  await page.locator('#new-game').click();
  await expect(page.locator('#restart')).toBeVisible();
  await page.locator('#cancel-restart').click();
  await page.getByRole('button', { name: 'Exit game fullscreen' }).click();
  await expect.poll(() => page.evaluate(() => document.fullscreenElement)).toBeNull();
  await expect(page.getByRole('button', { name: 'Enter game fullscreen' })).toHaveAttribute('aria-pressed', 'false');
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(true);
});


test('a human checkmate in fullscreen offers an immediate new game', async ({ page }) => {
  await seed(page, ['e4', 'e5', 'Qh5', 'Nc6', 'Bc4', 'Nf6']);
  await page.getByRole('radio', { name: 'Brown', exact: true }).check();
  await page.getByRole('button', { name: 'Enter game fullscreen' }).click();
  await move(page, 'Qxf7#');
  await expect(page.getByRole('heading', { name: 'Checkmate', exact: true })).toBeVisible();
  await expect(page.locator('#result-winner')).toContainText('You win.');
  await expect(page.locator('#result-new-game')).toBeFocused();
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeDisabled();
  await page.screenshot({ path: 'printscreens/checkmate-victory.png', fullPage: true });
  await page.locator('#result-new-game').click();
  await expect(page.locator('#match-result')).toBeHidden();
  await expect(page.locator('#restart')).not.toBeVisible();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await expect(page.getByRole('radio', { name: 'Brown', exact: true })).toBeChecked();
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeEnabled();
  expect(await page.evaluate(() => document.fullscreenElement?.id)).toBe('game-layout');
  await move(page, 'e4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
});

test('a CPU checkmate on mobile shows the result and restarts from its button', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await seed(page, ['f3', 'e5']);
  await move(page, 'g4');
  await expect(page.locator('#result-winner')).toContainText('The Castle Guardian wins.', { timeout: 15000 });
  await expect(page.getByRole('heading', { name: 'Checkmate', exact: true })).toBeInViewport();
  await expect(page.locator('#result-new-game')).toBeInViewport();
  await page.screenshot({ path: 'printscreens/checkmate-mobile.png', fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  await page.locator('#result-new-game').click();
  await expect(page.locator('#match-result')).toBeHidden();
  await expect(page.locator('#move-count')).toHaveText('0 MOVES');
  await expect(page.getByRole('textbox', { name: 'Move notation' })).toBeEnabled();
});

test('piece styles and backgrounds restyle the table without touching the match, and survive reload', async ({ page }) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await move(page, 'e4');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES', { timeout: 15000 });
  for (const [room, label, file] of [['greatHall', 'HOGWARTS · THE GREAT HALL', 'great-hall-room'], ['office', 'HOGWARTS · DUMBLEDORE’S OFFICE', 'dumbledore-office'], ['library', 'HOGWARTS · THE LIBRARY', 'library-room']]) {
    await page.getByLabel('BACKGROUND').selectOption(room);
    await expect(page.locator('#room-name')).toHaveText(label);
    await page.waitForTimeout(300);
    await page.screenshot({ path: `printscreens/background-${file}.png`, fullPage: true });
  }
  for (const style of ['marble', 'wood', 'steel', 'glass', 'plastic', 'stone']) {
    await page.getByLabel('PIECE STYLE').selectOption(style);
    await page.waitForTimeout(300);
    await page.screenshot({ path: `printscreens/style-${style}.png`, fullPage: true });
  }
  await page.getByLabel('BACKGROUND').selectOption('office');
  await page.getByLabel('PIECE STYLE').selectOption('glass');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  await page.reload();
  await expect(page.getByLabel('PIECE STYLE')).toHaveValue('glass');
  await expect(page.getByLabel('BACKGROUND')).toHaveValue('office');
  await expect(page.locator('#room-name')).toHaveText('HOGWARTS · DUMBLEDORE’S OFFICE');
  await expect(page.locator('#move-count')).toHaveText('2 MOVES');
  expect(errors).toEqual([]);
});

test('the fireplace crackles only while sound is on in the library', async ({ page }) => {
  await page.addInitScript(() => {
    const fire = { playing: 0 };
    Object.assign(window, { fireCheck: fire });
    const start = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (...args) {
      if (this.loop) fire.playing++;
      return start.apply(this, args);
    };
    const stop = AudioBufferSourceNode.prototype.stop;
    AudioBufferSourceNode.prototype.stop = function (...args) {
      if (this.loop) fire.playing--;
      return stop.apply(this, args);
    };
  });
  await page.route(/onlinesequencer/, route => route.abort());
  await page.goto('/');
  const playing = () => page.evaluate(() => (window as unknown as { fireCheck: { playing: number } }).fireCheck.playing);
  expect(await playing()).toBe(0);
  await page.getByRole('button', { name: 'Play library music and sound' }).click();
  await expect.poll(playing).toBe(1);
  await page.getByLabel('BACKGROUND').selectOption('greatHall');
  await expect.poll(playing).toBe(0);
  await page.getByLabel('BACKGROUND').selectOption('office');
  await expect.poll(playing).toBe(0);
  await page.getByLabel('BACKGROUND').selectOption('library');
  await expect.poll(playing).toBe(1);
  await page.getByRole('button', { name: 'Mute library music and sound' }).click();
  await expect.poll(playing).toBe(0);
});
