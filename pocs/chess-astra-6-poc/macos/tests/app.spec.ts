import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const executable = '/Applications/Wizards Gambit.app/Contents/MacOS/Wizards Gambit';
const root = resolve(__dirname, '..', '..');
const script = (name: string) => execFileSync(join(root, 'scripts', name), { encoding: 'utf8' });
let userData = '';

test.beforeAll(() => {
  const running = spawnSync('pgrep', ['-lx', 'Wizards Gambit'], { encoding: 'utf8' });
  if (running.status === 0) throw new Error(`Quit Wizards Gambit before running these tests: they start and stop the same game server the open app is using.\n${running.stdout}`);
});
test.beforeEach(() => { userData = mkdtempSync(join(tmpdir(), 'wizards-gambit-')); });
test.afterEach(() => rmSync(userData, { recursive: true, force: true }));

async function launch() {
  const app = await electron.launch({ executablePath: executable, env: { ...process.env, WIZARDS_GAMBIT_USER_DATA: userData } });
  const page = await app.firstWindow();
  return { app, page };
}

async function quit(app: ElectronApplication) {
  await Promise.all([app.waitForEvent('close', { timeout: 60000 }), app.evaluate(({ app }) => app.quit())]);
}

async function ready(page: Page) {
  await expect(page.locator('#scene canvas')).toBeVisible({ timeout: 90000 });
  await expect(page.locator('#mac-strip')).toBeVisible();
}

const clickMenu = (app: ElectronApplication, id: string) => app.evaluate(({ Menu }, id) => Menu.getApplicationMenu()!.getMenuItemById(id)!.click(), id);
const bounds = (app: ElectronApplication) => app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBounds());

test('opening the app boots the game server with start-all, and quitting runs stop-all', async () => {
  script('stop-all.sh');
  expect(script('status.sh')).toContain('DOWN');
  const { app, page } = await launch();
  await expect(page.locator('.services li'), 'the boot screen names every service it starts').toHaveCount(4);
  await expect(page.locator('[data-id="runtime"]')).toHaveAttribute('data-status', 'ready', { timeout: 30000 });
  await page.screenshot({ path: join(root, 'printscreens', 'macos-boot.png') });
  await expect(page.locator('#scene canvas'), 'the game appears once start-all reports the server up').toBeVisible({ timeout: 90000 });
  expect(script('status.sh')).toContain('UP');
  await expect(page.locator('#mac-strip')).toBeVisible();
  expect(await page.locator('#app').evaluate(app => app.getBoundingClientRect().top), 'the draggable title strip must not cover the game').toBeGreaterThanOrEqual(36);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(root, 'printscreens', 'macos-app.png') });
  await quit(app);
  expect(script('status.sh'), 'quitting must not leave the game server running').toContain('DOWN');
});

test('a second launch focuses the running app instead of opening another instance', async () => {
  const { app, page } = await launch();
  await ready(page);
  const second = spawn(executable, [], { env: { ...process.env, WIZARDS_GAMBIT_USER_DATA: userData } });
  const code = await new Promise<number | null>(done => second.on('exit', done));
  expect(code).toBe(0);
  expect(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().length)).toBe(1);
  expect(script('status.sh'), 'the second launch must not stop the first instance’s server').toContain('UP');
  await quit(app);
});

test('search goes anywhere, and the shortcut sheet groups, filters, fits and closes', async () => {
  const { app, page } = await launch();
  await ready(page);
  const accelerators = await app.evaluate(({ Menu }) => Object.fromEntries(['search', 'shortcuts', 'room-library', 'room-greatHall', 'room-office', 'fullscreen', 'capture', 'quit'].map(id => [id, Menu.getApplicationMenu()!.getMenuItemById(id)!.accelerator])));
  expect(accelerators).toEqual({ search: 'CmdOrCtrl+K', shortcuts: 'CmdOrCtrl+/', 'room-library': 'CmdOrCtrl+1', 'room-greatHall': 'CmdOrCtrl+2', 'room-office': 'CmdOrCtrl+3', fullscreen: 'CmdOrCtrl+Shift+Enter', capture: 'CmdOrCtrl+P', quit: 'CmdOrCtrl+Q' });

  await clickMenu(app, 'search');
  const search = page.getByRole('searchbox', { name: 'Search anything' });
  await expect(search).toBeFocused();
  await search.fill('great hall');
  await page.screenshot({ path: join(root, 'printscreens', 'macos-search.png') });
  await search.press('Enter');
  await expect(page.locator('#room-name')).toHaveText('HOGWARTS · THE GREAT HALL');
  await expect(page.locator('#mac-palette')).toBeHidden();
  await clickMenu(app, 'room-office');
  await expect(page.locator('#room-name')).toHaveText('HOGWARTS · DUMBLEDORE’S OFFICE');

  await clickMenu(app, 'shortcuts');
  const filter = page.getByRole('searchbox', { name: 'Keyboard shortcuts' });
  await expect(filter).toBeFocused();
  const groups = page.locator('#mac-shortcuts .mac-group');
  await expect(groups).toHaveCount(6);
  expect(await groups.evaluateAll(boxes => new Set(boxes.map(box => getComputedStyle(box.querySelector('h3')!).color)).size), 'each feature area has its own color').toBe(6);
  await expect(groups.locator('h3 svg')).toHaveCount(6);
  const fit = await page.locator('#mac-shortcuts .mac-card').evaluate(card => card.getBoundingClientRect().height <= innerHeight * 0.92);
  expect(fit).toBe(true);
  await page.screenshot({ path: join(root, 'printscreens', 'macos-shortcuts.png') });
  await filter.fill('zoom');
  await expect(page.locator('#mac-shortcuts .mac-shortcut')).toHaveCount(2);
  await expect(page.locator('#mac-shortcuts .mac-count')).toHaveText('2 of 24 shortcuts');
  await filter.fill('rooms');
  await expect(page.locator('#mac-shortcuts .mac-shortcut'), 'matching a group title keeps the whole group').toHaveCount(3);
  await filter.fill('xyzzy');
  await expect(page.locator('#mac-shortcuts .mac-empty')).toHaveText('No shortcuts match “xyzzy”.');
  await filter.press('Escape');
  await expect(filter).toHaveValue('');
  await expect(page.locator('#mac-shortcuts')).toBeVisible();
  await filter.press('Escape');
  await expect(page.locator('#mac-shortcuts')).toBeHidden();
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(600, 800));
  await clickMenu(app, 'shortcuts');
  await expect.poll(() => page.locator('#mac-shortcuts .mac-groups').evaluate(list => getComputedStyle(list).columnCount)).toBe('1');
  await quit(app);
});

test('the window remembers where it was, and double-clicking the title bar fills the screen and restores', async () => {
  const first = await launch();
  await ready(first.page);
  await first.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setBounds({ x: 120, y: 140, width: 1000, height: 760 }));
  await first.page.waitForTimeout(500);
  const placed = await bounds(first.app);
  await first.page.locator('#mac-strip').dblclick();
  const workArea = await first.app.evaluate(({ BrowserWindow, screen }) => screen.getDisplayMatching(BrowserWindow.getAllWindows()[0].getBounds()).workArea);
  await expect.poll(() => bounds(first.app)).toEqual(workArea);
  await first.page.locator('#mac-strip').dblclick();
  await expect.poll(() => bounds(first.app)).toEqual(placed);
  await quit(first.app);
  const second = await launch();
  expect(await bounds(second.app), 'reopening restores the exact position and size').toEqual(placed);
  await ready(second.page);
  await clickMenu(second.app, 'fullscreen');
  await expect.poll(() => second.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen()), { timeout: 10000 }).toBe(true);
  await quit(second.app);
  const third = await launch();
  await expect.poll(() => third.app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen()), { message: 'full screen is restored too', timeout: 10000 }).toBe(true);
  await ready(third.page);
  await quit(third.app);
});

test('cut, copy and paste work in the move field through the Edit menu', async () => {
  const { app, page } = await launch();
  await ready(page);
  const roles = await app.evaluate(({ Menu }) => Menu.getApplicationMenu()!.items.find(item => item.label === 'Edit')!.submenu!.items.map(item => item.role));
  expect(roles).toEqual(expect.arrayContaining(['cut', 'copy', 'paste', 'selectall']));
  const saved = await app.evaluate(({ clipboard }) => clipboard.readText());
  const input = page.getByRole('textbox', { name: 'Move notation' });
  await input.fill('e2e4');
  await input.focus();
  const run = (action: 'selectAll' | 'cut' | 'paste') => app.evaluate(({ BrowserWindow }, action) => BrowserWindow.getAllWindows()[0].webContents[action](), action);
  await run('selectAll');
  await run('cut');
  await expect(input).toHaveValue('');
  expect(await app.evaluate(({ clipboard }) => clipboard.readText())).toBe('e2e4');
  await run('paste');
  await expect(input).toHaveValue('e2e4');
  await app.evaluate(({ clipboard }, text) => clipboard.writeText(text), saved);
  await quit(app);
});
