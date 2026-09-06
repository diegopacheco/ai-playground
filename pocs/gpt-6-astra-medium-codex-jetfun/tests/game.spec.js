import { test, expect } from '@playwright/test';
async function ready(page) {
  await page.goto('/');
  await page.waitForFunction(() => window.jetfun?.renderer.info.render.calls > 0);
}
test('renders the bay, selects both riders, and opens accessible route and help dialogs', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await ready(page);
  await expect(page.getByRole('heading', { name: 'Make waves.' })).toBeVisible();
  await expect(page.locator('#load-error')).toBeHidden();
  await page.screenshot({ path: 'printscreens/desktop.png', fullPage: true });
  await page.locator('[data-rider="man"]').click();
  expect(await page.evaluate(() => jetfun.state.rider)).toBe('man');
  await page.locator('[data-rider="woman"]').click();
  expect(await page.evaluate(() => jetfun.state.rider)).toBe('woman');
  await page.locator('#routes-nav').click();
  await expect(page.locator('.route-list li')).toHaveCount(7);
  await page.screenshot({ path: 'printscreens/route.png', fullPage: true });
  await page.keyboard.press('Escape');
  await page.locator('#help').click();
  await expect(page.getByRole('dialog')).toContainText('Brake and reverse');
  await page.screenshot({ path: 'printscreens/guide.png', fullPage: true });
  await page.keyboard.press('Escape');
  await page.locator('#sound').click();
  await expect(page.locator('#sound')).toHaveAttribute('aria-label', 'Mute sound');
  await page.locator('#sound').click();
  expect(errors).toEqual([]);
});
test('keyboard acceleration, steering, boost, pause, and restart control the race', async ({ page }) => {
  await ready(page);
  await page.locator('#start').click();
  await page.waitForFunction(() => jetfun.state.phase === 'racing');
  await page.keyboard.down('ArrowUp');
  await page.waitForFunction(() => jetfun.state.speed > 10);
  await page.keyboard.down('ArrowRight');
  await page.waitForFunction(() => jetfun.state.x > 1);
  await page.keyboard.up('ArrowRight');
  await page.keyboard.down('Shift');
  await page.waitForFunction(() => jetfun.state.boost < 95);
  await page.keyboard.up('Shift');
  await page.keyboard.up('ArrowUp');
  await page.keyboard.press('p');
  expect(await page.evaluate(() => jetfun.state.phase)).toBe('paused');
  const elapsed = await page.evaluate(() => jetfun.state.elapsed);
  await page.evaluate(() => jetfun.updateRace(2));
  expect(await page.evaluate(() => jetfun.state.elapsed)).toBe(elapsed);
  await page.screenshot({ path: 'printscreens/race.png', fullPage: true });
  await page.keyboard.press('p');
  expect(await page.evaluate(() => jetfun.state.phase)).toBe('racing');
  await page.keyboard.press('r');
  const reset = await page.evaluate(() => ({ gate: jetfun.state.gate, time: jetfun.state.time, speed: jetfun.state.speed }));
  expect(reset).toEqual({ gate: 0, time: 180, speed: 0 });
});
test('ordered checkpoint detection, shoreline collision, timeout, finish, and saved best', async ({ page }) => {
  await ready(page);
  const swept = await page.evaluate(() => jetfun.checkpointHit({x:0,z:30},{x:0,z:-30},{x:0,z:0}));
  expect(swept).toBe(true);
  await page.evaluate(() => { jetfun.reset(); Object.assign(jetfun.state, {phase:'racing',x:0,z:-410}); jetfun.updateRace(.01); });
  expect(await page.evaluate(() => jetfun.state.gate)).toBe(0);
  await page.evaluate(() => { Object.assign(jetfun.state,{x:-47.9,z:95,heading:Math.PI/2,speed:20});jetfun.updateRace(.1); });
  expect(await page.evaluate(() => jetfun.state.x)).toBeGreaterThanOrEqual(-48);
  expect(await page.evaluate(() => jetfun.state.speed)).toBeLessThan(0);
  await page.evaluate(() => { Object.assign(jetfun.state,{elapsed:179.99});jetfun.updateRace(.1); });
  expect(await page.evaluate(() => jetfun.state.phase)).toBe('failed');
  await expect(page.getByRole('dialog')).toContainText('Time’s up');
  await page.locator('#ride-again').click();
  await page.evaluate(() => {jetfun.state.phase='racing';for(const p of jetfun.checkpoints){Object.assign(jetfun.state,{x:p.x,z:p.z,speed:0});jetfun.updateRace(1);}});
  expect(await page.evaluate(() => jetfun.state.phase)).toBe('finished');
  await expect(page.getByRole('dialog')).toContainText('You made it to Sunset Beach');
  await page.screenshot({ path: 'printscreens/finish.png', fullPage: true });
  expect(await page.evaluate(() => Number(localStorage.getItem('jetfun-best')))).toBeGreaterThan(0);
  await page.reload();
  await page.waitForFunction(() => window.jetfun);
  await expect(page.locator('#best')).not.toContainText('—');
});
test('mobile detection, responsive layout, and touch acceleration', async ({ browser }) => {
  const context = await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:1});
  const page = await context.newPage();
  await ready(page);
  await expect(page.locator('body')).toHaveClass('touch-device');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({path:'printscreens/mobile.png',fullPage:true});
  await page.locator('#start').tap();
  await page.waitForFunction(() => jetfun.state.phase === 'racing');
  await page.locator('[data-key="ArrowUp"]').dispatchEvent('pointerdown',{pointerId:1,pointerType:'touch'});
  await page.waitForFunction(() => jetfun.state.speed > 5);
  await page.locator('[data-key="ArrowUp"]').dispatchEvent('pointerup',{pointerId:1,pointerType:'touch'});
  await page.locator('#pause').tap();
  expect(await page.evaluate(() => jetfun.state.phase)).toBe('paused');
  await context.close();
});
test('the complete course is reachable using throttle and steering without moving the rider directly', async ({ page }) => {
  await ready(page);
  const result = await page.evaluate(() => {
    jetfun.reset();
    jetfun.updateRace(3.1);
    const held = new Set();
    function hold(code, active) {
      if (active === held.has(code)) return;
      window.dispatchEvent(new KeyboardEvent(active ? 'keydown' : 'keyup', {code}));
      if (active) held.add(code); else held.delete(code);
    }
    for (let i=0;i<3600 && jetfun.state.phase==='racing';i++) {
      const state=jetfun.state, point=jetfun.checkpoints[state.gate];
      const desired=Math.atan2(-(point.x-state.x),-(point.z-state.z));
      const error=Math.atan2(Math.sin(desired-state.heading),Math.cos(desired-state.heading));
      hold('KeyW',true);
      hold('KeyA',error>.04);
      hold('KeyD',error<-.04);
      jetfun.updateRace(.05);
    }
    for(const code of held) window.dispatchEvent(new KeyboardEvent('keyup',{code}));
    return {phase:jetfun.state.phase,gate:jetfun.state.gate,elapsed:jetfun.state.elapsed};
  });
  expect(result.phase).toBe('finished');
  expect(result.gate).toBe(7);
  expect(result.elapsed).toBeLessThan(180);
  await page.evaluate(() => new Promise(resolve => {let frames=0;function next(){if(++frames===40)resolve();else requestAnimationFrame(next);}requestAnimationFrame(next);}));
  await page.screenshot({path:'printscreens/finish.png',fullPage:true});
});
